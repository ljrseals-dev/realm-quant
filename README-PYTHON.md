# Realm of Quant — Python Workbench

This is the first executable application layer for the Realm of Quant research.
It operates on real ONNX files and audio; it is not a prompt runner or a review
pipeline.

## What it does

- Inspects ONNX model input/output contracts through ONNX Runtime.
- Produces FP16 ONNX models (optional `onnxconverter-common` dependency).
- Produces dynamic INT8 weight-quantized ONNX models for selected operators.
- Produces packed 2-bit ternary weight sidecars with per-output-channel scales,
  plus a dequantized ONNX model that ONNX Runtime can execute.
- Encodes WAV audio to DAC code tensors, decodes code tensors to WAV, and runs
  full encoder/decoder round trips.
- Compares reference and candidate models on identical saved inputs, reporting
  shape, finiteness, maximum absolute error, RMSE, relative L2 error, and cosine
  similarity.

**Important ternary boundary:** the packed ternary sidecar is a real compact
weight representation, but the generated ONNX model reconstructs those weights
as floating point for compatibility. This release does not claim native
2-bit arithmetic or a speedup from ternary execution. A packed-kernel backend
is a subsequent implementation milestone.

## Install

Python 3.10+:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev,fp16]"
```

For INT8, the base ONNX Runtime dependency is sufficient. Install SciPy for
higher-quality resampling when source WAV files are not 24 kHz:

```bash
pip install -e ".[resample]"
```

## Use with the public DAC 24 kHz ONNX exports

Download the FP32 or FP16 encoder/decoder files from
[onnx-community/dac_24khz-ONNX](https://huggingface.co/onnx-community/dac_24khz-ONNX)
and keep the model files outside Git.

Inspect:

```bash
realm-quant inspect encoder_model.onnx
realm-quant inspect decoder_model_fp16.onnx
```

Create quantized models:

```bash
realm-quant quantize decoder_model.onnx --mode fp16 --output build/decoder.fp16.onnx
realm-quant quantize encoder_model.onnx --mode int8 --op-types MatMul,Gemm --output build/encoder.int8.onnx
realm-quant quantize decoder_model.onnx --mode ternary --threshold-ratio 0.7 --output build/decoder.ternary.onnx
```

Ternary output consists of:

- `build/decoder.ternary.onnx`: ONNX-compatible model with reconstructed float weights.
- `build/decoder.ternary.npz`: packed 2-bit symbols and per-output-channel scales.
- `build/decoder.ternary.json`: tensor map, packing format, size accounting, and reconstruction errors.

Run a real audio round trip:

```bash
realm-quant roundtrip encoder_model.onnx decoder_model_fp16.onnx input.wav --output build/reconstructed.wav
```

Or encode/decode separately:

```bash
realm-quant encode encoder_model.onnx input.wav --codes build/codes.npy
realm-quant decode decoder_model_fp16.onnx build/codes.npy --output build/decoded.wav
```

## Numerical comparison

Save the exact model input tensors in an NPZ archive, keyed by ONNX input name.
Then compare reference and candidate:

```bash
realm-quant compare reference.onnx candidate.onnx --feeds inputs.npz --report build/comparison.json
```

For a decoder, the NPZ should contain an `audio_codes` array with shape
`[1, 32, frames]` and integer dtype. For an encoder, it should contain the
waveform input with shape `[1, 1, samples]` and float32 dtype.

## Verification

```bash
pytest
```

The unit tests cover ternary symbol packing and quantizer invariants. Model-level
validation requires the actual ONNX model files and is intentionally not
represented as passing until those files are supplied and the commands are run.

## Current scope

This is the Python execution path. It does not compile Cmajor, generate WASM,
or claim that the Cmajor emitter's operator/type limitations have been fixed.
