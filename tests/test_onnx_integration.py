from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from realm_quant.quantization import ternary_quantize_model, unpack_2bit
from realm_quant.runtime import compare_models


def _make_matmul_model(path: Path) -> np.ndarray:
    weights = np.array(
        [
            [0.25, -0.5, 1.0],
            [1.5, 0.2, -0.7],
            [-0.4, 0.9, 0.3],
            [0.8, -1.2, 0.6],
        ],
        dtype=np.float32,
    )
    input_info = helper.make_tensor_value_info("features", TensorProto.FLOAT, [None, 4])
    output_info = helper.make_tensor_value_info("scores", TensorProto.FLOAT, [None, 3])
    weight_tensor = numpy_helper.from_array(weights, name="projection.weight")
    node = helper.make_node("MatMul", ["features", "projection.weight"], ["scores"])
    graph = helper.make_graph([node], "tiny_projection", [input_info], [output_info], [weight_tensor])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return weights


def test_ternary_model_is_executable_and_comparable(tmp_path):
    source = tmp_path / "reference.onnx"
    candidate = tmp_path / "ternary.onnx"
    weights = _make_matmul_model(source)
    result = ternary_quantize_model(source, candidate, threshold_ratio=0.6)

    assert Path(result["model"]).is_file()
    assert Path(result["sidecar"]).is_file()
    assert Path(result["manifest"]).is_file()
    assert result["tensor_count"] == 1

    with np.load(result["sidecar"], allow_pickle=False) as packed:
        symbols = unpack_2bit(packed["tensor_0_symbols"], weights.size).reshape(weights.shape)
        scales = packed["tensor_0_scales"]
    assert scales.shape == (weights.shape[1],)
    assert set(np.unique(symbols)).issubset({0, 1, 2})

    input_data = np.array([[0.2, -0.1, 0.7, 1.0]], dtype=np.float32)
    reference_session = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"])
    candidate_session = ort.InferenceSession(str(candidate), providers=["CPUExecutionProvider"])
    reference_output = reference_session.run(None, {"features": input_data})[0]
    candidate_output = candidate_session.run(None, {"features": input_data})[0]
    assert reference_output.shape == candidate_output.shape == (1, 3)
    assert np.isfinite(candidate_output).all()

    feeds = tmp_path / "feeds.npz"
    np.savez(feeds, features=input_data)
    report = compare_models(source, candidate, feeds)
    assert report["outputs"][0]["shape_match"] is True
    assert report["outputs"][0]["finite"] is True
