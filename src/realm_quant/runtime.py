"""ONNX Runtime inference, codec audio I/O, and numerical comparison."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np


def _ort_session(model_path: str | Path, threads: int = 1):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = max(1, threads)
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])


def inspect_model(model_path: str | Path) -> dict[str, Any]:
    session = _ort_session(model_path)
    return {
        "model": str(model_path),
        "providers": session.get_providers(),
        "inputs": [
            {"name": x.name, "shape": x.shape, "type": x.type}
            for x in session.get_inputs()
        ],
        "outputs": [
            {"name": x.name, "shape": x.shape, "type": x.type}
            for x in session.get_outputs()
        ],
    }


def compare_models(
    reference_path: str | Path,
    candidate_path: str | Path,
    feeds_path: str | Path,
    threads: int = 1,
) -> dict[str, Any]:
    """Run identical named inputs through two models and report output error."""
    reference = _ort_session(reference_path, threads)
    candidate = _ort_session(candidate_path, threads)
    with np.load(feeds_path, allow_pickle=False) as data:
        feeds = {name: data[name] for name in data.files}

    ref_names = {x.name for x in reference.get_inputs()}
    cand_names = {x.name for x in candidate.get_inputs()}
    if ref_names != cand_names:
        raise ValueError(f"input-name mismatch: reference={sorted(ref_names)}, candidate={sorted(cand_names)}")
    missing = ref_names - feeds.keys()
    if missing:
        raise ValueError(f"feed archive is missing inputs: {sorted(missing)}")

    ref_outputs = reference.run(None, {k: v for k, v in feeds.items() if k in ref_names})
    cand_outputs = candidate.run(None, {k: v for k, v in feeds.items() if k in cand_names})
    if len(ref_outputs) != len(cand_outputs):
        raise ValueError("models return different numbers of outputs")

    results = []
    for meta, left, right in zip(reference.get_outputs(), ref_outputs, cand_outputs):
        a = np.asarray(left, dtype=np.float64)
        b = np.asarray(right, dtype=np.float64)
        if a.shape != b.shape:
            results.append({"output": meta.name, "reference_shape": list(a.shape), "candidate_shape": list(b.shape), "shape_match": False})
            continue
        delta = a - b
        denom = float(np.linalg.norm(a.ravel()))
        ref_rms = float(np.sqrt(np.mean(np.square(a)))) if a.size else 0.0
        rmse = float(np.sqrt(np.mean(np.square(delta)))) if a.size else 0.0
        dot = float(np.dot(a.ravel(), b.ravel()))
        cosine_denom = float(np.linalg.norm(a.ravel()) * np.linalg.norm(b.ravel()))
        results.append({
            "output": meta.name,
            "shape": list(a.shape),
            "shape_match": True,
            "finite": bool(np.isfinite(a).all() and np.isfinite(b).all()),
            "max_abs_error": float(np.max(np.abs(delta))) if a.size else 0.0,
            "rmse": rmse,
            "relative_l2_error": float(np.linalg.norm(delta.ravel()) / denom) if denom else None,
            "rmse_over_reference_rms": rmse / ref_rms if ref_rms else None,
            "cosine_similarity": dot / cosine_denom if cosine_denom else None,
        })
    return {
        "reference": str(reference_path),
        "candidate": str(candidate_path),
        "feeds": str(feeds_path),
        "outputs": results,
    }


def _load_audio(path: str | Path, sample_rate: int = 24000) -> np.ndarray:
    import soundfile as sf

    audio, source_rate = sf.read(str(path), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if source_rate != sample_rate:
        try:
            from scipy.signal import resample_poly
            from math import gcd

            divisor = gcd(int(source_rate), sample_rate)
            mono = resample_poly(mono, sample_rate // divisor, source_rate // divisor).astype(np.float32)
        except ImportError:
            old_x = np.arange(mono.size, dtype=np.float64) / source_rate
            new_n = int(round(mono.size * sample_rate / source_rate))
            new_x = np.arange(new_n, dtype=np.float64) / sample_rate
            mono = np.interp(new_x, old_x, mono).astype(np.float32)
    return np.clip(mono, -1.0, 1.0)


def _write_audio(path: str | Path, audio: np.ndarray, sample_rate: int = 24000) -> None:
    import soundfile as sf

    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if not np.isfinite(samples).all():
        raise ValueError("model output contains NaN or infinity")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(samples, -1.0, 1.0), sample_rate, subtype="PCM_16")


def encode_audio(
    encoder_path: str | Path,
    audio_path: str | Path,
    codes_path: str | Path,
    threads: int = 1,
) -> dict[str, Any]:
    session = _ort_session(encoder_path, threads)
    audio = _load_audio(audio_path)
    input_meta = session.get_inputs()[0]
    tensor = audio.reshape(1, 1, -1).astype(np.float32)
    start = time.perf_counter()
    outputs = session.run(None, {input_meta.name: tensor})
    elapsed = time.perf_counter() - start

    chosen = None
    chosen_meta = None
    for meta, value in zip(session.get_outputs(), outputs):
        array = np.asarray(value)
        if array.ndim == 3 and array.shape[1] == 32 and array.dtype.kind in "iu":
            chosen, chosen_meta = array, meta
            break
    if chosen is None:
        raise RuntimeError("encoder produced no rank-3 integer [batch, 32, frames] code tensor")
    destination = Path(codes_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.save(destination, chosen)
    duration = audio.size / 24000
    return {
        "codes": str(destination),
        "output_name": chosen_meta.name,
        "shape": list(chosen.shape),
        "audio_seconds": duration,
        "inference_seconds": elapsed,
        "realtime_factor": elapsed / duration if duration else None,
    }


def decode_codes(
    decoder_path: str | Path,
    codes_path: str | Path,
    output_path: str | Path,
    threads: int = 1,
) -> dict[str, Any]:
    session = _ort_session(decoder_path, threads)
    codes = np.load(codes_path, allow_pickle=False)
    if codes.ndim != 3 or codes.shape[1] != 32:
        raise ValueError(f"expected codes shaped [batch, 32, frames], got {codes.shape}")
    input_meta = session.get_inputs()[0]
    expected = input_meta.type
    if expected == "tensor(int64)":
        codes = codes.astype(np.int64, copy=False)
    elif expected == "tensor(int32)":
        codes = codes.astype(np.int32, copy=False)
    start = time.perf_counter()
    outputs = session.run(None, {input_meta.name: codes})
    elapsed = time.perf_counter() - start
    audio = np.asarray(outputs[0]).squeeze()
    if audio.ndim != 1:
        raise RuntimeError(f"decoder output is not mono waveform: shape {audio.shape}")
    _write_audio(output_path, audio)
    duration = audio.size / 24000
    return {
        "audio": str(output_path),
        "samples": int(audio.size),
        "sample_rate": 24000,
        "audio_seconds": duration,
        "inference_seconds": elapsed,
        "realtime_factor": elapsed / duration if duration else None,
    }


def roundtrip_audio(
    encoder_path: str | Path,
    decoder_path: str | Path,
    audio_path: str | Path,
    output_path: str | Path,
    threads: int = 1,
) -> dict[str, Any]:
    codes_path = Path(output_path).with_suffix(".codes.npy")
    encoded = encode_audio(encoder_path, audio_path, codes_path, threads)
    decoded = decode_codes(decoder_path, codes_path, output_path, threads)
    return {"encode": encoded, "decode": decoded}


def write_json(data: dict[str, Any], output: str | Path | None = None) -> None:
    rendered = json.dumps(data, indent=2, allow_nan=False)
    if output:
        Path(output).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
