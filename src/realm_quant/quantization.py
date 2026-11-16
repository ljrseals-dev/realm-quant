"""ONNX weight quantization with explicit, inspectable artifacts.

Ternary mode stores packed 2-bit symbols plus per-output-channel scales, and
also writes a dequantized float ONNX model so ordinary ONNX Runtime can execute
the quantized weights without custom kernels. This is weight-only quantization:
it does not claim integer/ternary arithmetic in the execution backend.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def pack_2bit(codes: np.ndarray) -> np.ndarray:
    """Pack codes 0..3, four values per byte, least-significant pair first."""
    values = np.asarray(codes, dtype=np.uint8).reshape(-1)
    if np.any(values > 3):
        raise ValueError("2-bit symbols must be in [0, 3]")
    padded = np.pad(values, (0, (-values.size) % 4))
    groups = padded.reshape(-1, 4)
    return (
        groups[:, 0]
        | (groups[:, 1] << 2)
        | (groups[:, 2] << 4)
        | (groups[:, 3] << 6)
    ).astype(np.uint8)


def unpack_2bit(packed: np.ndarray, count: int) -> np.ndarray:
    """Unpack the first count symbols from a 2-bit byte stream."""
    if count < 0 or count > np.asarray(packed).size * 4:
        raise ValueError("count exceeds packed symbol capacity")
    data = np.asarray(packed, dtype=np.uint8).reshape(-1)
    out = np.empty(data.size * 4, dtype=np.uint8)
    out[0::4] = data & 0x03
    out[1::4] = (data >> 2) & 0x03
    out[2::4] = (data >> 4) & 0x03
    out[3::4] = (data >> 6) & 0x03
    return out[:count]


def _ternarize(
    array: np.ndarray,
    threshold_ratio: float,
    axis: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return symbols, per-output-channel scales, and reconstructed weights."""
    weights = np.asarray(array, dtype=np.float32)
    if weights.ndim < 2:
        raise ValueError("ternary weight tensors must have rank >= 2")
    if not np.isfinite(weights).all():
        raise ValueError("weight tensor contains NaN or infinity")
    if axis < 0:
        axis += weights.ndim
    if not 0 <= axis < weights.ndim:
        raise ValueError(f"scale axis {axis} is invalid for rank {weights.ndim}")
    moved = np.moveaxis(weights, axis, 0)
    rows = moved.reshape(moved.shape[0], -1)
    magnitudes = np.abs(rows)
    thresholds = magnitudes.mean(axis=1, keepdims=True) * threshold_ratio
    active = (magnitudes >= thresholds) & (magnitudes > 0)
    counts = active.sum(axis=1, keepdims=True)
    scales = np.divide(
        (magnitudes * active).sum(axis=1, keepdims=True),
        counts,
        out=np.zeros((rows.shape[0], 1), dtype=np.float32),
        where=counts != 0,
    )
    symbols = np.zeros(rows.shape, dtype=np.uint8)
    symbols[(rows > 0) & active] = 1
    symbols[(rows < 0) & active] = 2
    reconstructed = np.where(symbols == 1, scales, np.where(symbols == 2, -scales, 0.0))
    symbols = np.moveaxis(symbols.reshape(moved.shape), 0, axis)
    restored = np.moveaxis(reconstructed.reshape(moved.shape), 0, axis)
    return symbols, scales.reshape(-1), restored


def _initializer_output_axes(model: Any) -> dict[str, int]:
    """Infer the output-channel axis from the ONNX operator consuming each weight."""
    import onnx

    axes: dict[str, int] = {}
    for node in model.graph.node:
        weight_index = None
        axis = 0
        attrs = {item.name: onnx.helper.get_attribute_value(item) for item in node.attribute}
        if node.op_type in ("Conv", "QLinearConv"):
            weight_index, axis = (3, 0) if node.op_type == "QLinearConv" else (1, 0)
        elif node.op_type == "ConvTranspose":
            weight_index, axis = 1, 1
        elif node.op_type in ("MatMul", "MatMulInteger"):
            weight_index, axis = 1, -1
        elif node.op_type == "QLinearMatMul":
            weight_index, axis = 3, -1
        elif node.op_type == "Gemm":
            weight_index = 1
            axis = -2 if int(attrs.get("transB", 0)) else -1

        if weight_index is not None and len(node.input) > weight_index:
            name = node.input[weight_index]
            if name:
                axes.setdefault(name, axis)
    return axes

def ternary_quantize_model(
    model_path: str | Path,
    output_path: str | Path,
    threshold_ratio: float = 0.7,
) -> dict[str, Any]:
    """Write an executable dequantized ONNX model and packed ternary sidecar."""
    if not 0.0 <= threshold_ratio <= 2.0:
        raise ValueError("threshold_ratio must be between 0 and 2")

    import onnx
    from onnx import numpy_helper

    source = Path(model_path)
    destination = Path(output_path)
    model = onnx.load(str(source), load_external_data=True)
    packed_arrays: dict[str, np.ndarray] = {}
    manifest: dict[str, Any] = {
        "format": "realm-quant-ternary-v1",
        "source_model": source.name,
        "executable_model": destination.name,
        "threshold_ratio": threshold_ratio,
        "symbol_encoding": {"0": "zero", "1": "positive", "2": "negative", "3": "reserved"},
        "packing": "four 2-bit symbols per byte; least-significant pair first",
        "execution_note": (
            "The ONNX model contains dequantized float weights for compatibility. "
            "Packed symbols are preserved in the sidecar; this is not native ternary arithmetic."
        ),
        "tensors": [],
    }
    output_axes = _initializer_output_axes(model)
    original_bytes = 0
    packed_bytes = 0
    total_values = 0
    active_values = 0

    for index, tensor in enumerate(model.graph.initializer):
        array = numpy_helper.to_array(tensor)
        if array.dtype.kind != "f" or array.ndim < 2 or array.size < 16:
            continue
        scale_axis = output_axes.get(tensor.name, 0)
        symbols, scales, reconstructed = _ternarize(array, threshold_ratio, scale_axis)
        packed = pack_2bit(symbols)
        key = f"tensor_{index}"
        packed_arrays[f"{key}_symbols"] = packed
        packed_arrays[f"{key}_scales"] = scales.astype(np.float32)
        tensor.CopyFrom(numpy_helper.from_array(reconstructed.astype(array.dtype), tensor.name))

        item = {
            "name": tensor.name,
            "shape": list(array.shape),
            "symbol_key": f"{key}_symbols",
            "scale_key": f"{key}_scales",
            "symbol_count": int(array.size),
            "scale_axis": scale_axis,
            "scale_count": int(scales.size),
            "original_dtype": str(array.dtype),
            "original_bytes": int(array.nbytes),
            "packed_bytes": int(packed.nbytes + scales.nbytes),
            "nonzero_fraction": float(np.count_nonzero(symbols) / symbols.size),
            "rmse": float(np.sqrt(np.mean(np.square(array.astype(np.float32) - reconstructed)))),
        }
        manifest["tensors"].append(item)
        original_bytes += array.nbytes
        packed_bytes += packed.nbytes + scales.nbytes
        total_values += symbols.size
        active_values += int(np.count_nonzero(symbols))

    if not manifest["tensors"]:
        raise ValueError("no eligible floating-point weight tensors found (rank >= 2, size >= 16)")

    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, str(destination))
    sidecar = destination.with_suffix(".ternary.npz")
    np.savez_compressed(sidecar, **packed_arrays)
    manifest["summary"] = {
        "tensor_count": len(manifest["tensors"]),
        "weight_values": total_values,
        "nonzero_fraction": active_values / total_values,
        "original_weight_bytes": original_bytes,
        "packed_symbols_and_scales_bytes": packed_bytes,
        "packed_ratio": packed_bytes / original_bytes if original_bytes else None,
        "note": "Ratio excludes ONNX graph, biases, metadata, and NPZ container overhead.",
    }
    manifest_path = destination.with_suffix(".ternary.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {
        "model": str(destination),
        "sidecar": str(sidecar),
        "manifest": str(manifest_path),
        **manifest["summary"],
    }


def quantize_model(
    model_path: str | Path,
    output_path: str | Path,
    mode: str,
    threshold_ratio: float = 0.7,
    op_types: tuple[str, ...] = ("MatMul", "Gemm"),
) -> dict[str, Any]:
    """Quantize an ONNX model as fp16, dynamic int8, or packed ternary weights."""
    source = Path(model_path)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if mode == "ternary":
        return ternary_quantize_model(source, destination, threshold_ratio)

    if mode == "fp16":
        try:
            from onnxconverter_common import float16
        except ImportError as exc:
            raise RuntimeError(
                "FP16 conversion requires the optional dependency: "
                "pip install 'realm-quant[fp16]'"
            ) from exc
        import onnx

        model = onnx.load(str(source), load_external_data=True)
        converted = float16.convert_float_to_float16(model, keep_io_types=True)
        onnx.save_model(converted, str(destination))
        return {"model": str(destination), "mode": "fp16", "note": "I/O tensor types preserved."}

    if mode == "int8":
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic
        except ImportError as exc:
            raise RuntimeError("INT8 conversion requires onnxruntime.") from exc
        quantize_dynamic(
            model_input=str(source),
            model_output=str(destination),
            weight_type=QuantType.QInt8,
            per_channel=True,
            op_types_to_quantize=list(op_types),
        )
        return {
            "model": str(destination),
            "mode": "dynamic-int8",
            "op_types": list(op_types),
            "note": "Dynamic weight quantization; operators outside op_types remain unchanged.",
        }

    raise ValueError(f"unsupported quantization mode: {mode}")
