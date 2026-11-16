# ONNX Runtime ↔ Cmajor native adapter

This bridge allows a Cmajor program hosted by a native LLVM engine to call an
ONNX Runtime model through a small C ABI. It does not translate ONNX operators
into Cmajor processors: ONNX Runtime remains the graph executor. Cmajor is the
host-side adapter/control layer.

## Contract

- One ONNX input and one ONNX output per session.
- Explicit positive input dimensions; row-major contiguous tensors.
- Input ABI is float32. The bridge converts to ONNX float16, float32, int32,
  or int64. Integer conversion truncates toward zero.
- Output is converted to float32 and its dimensions are returned.
- Dynamic output dimensions are accepted only when ONNX Runtime resolves them
  during execution.
- The caller owns buffers and must provide sufficient output capacity.
- The registry serializes calls. Do not call inference from a hard real-time
  callback: it allocates, locks, and may block. Use a worker and bounded SPSC
  queue for audio integration.
- Handle 0 means open failure; run returns -1 on failure. Retrieve diagnostics
  with rq_ort_last_error.
- Native LLVM host only. Cmajor external functions are not a browser/WASM ABI.

## Build

Install the ONNX Runtime C/C++ SDK (headers and library), then run:

    cmake -S bridge -B build/bridge -DONNXRUNTIME_ROOT=/path/to/onnxruntime
    cmake --build build/bridge --config Release

Make the resulting shared library visible to the native host. Link the Cmajor
LLVM engine with its exported C ABI symbols and load
bridge/onnx_runtime.cmajor into the Cmajor program.

## Lifecycle

1. Call rq_ort_open(path, threads) outside audio processing.
2. Determine the model's input shape and allocate buffers off-thread.
3. Call rq_ort_run_f32 from a worker, never from the realtime callback.
4. Consume returned output dimensions and float32 values.
5. Call rq_ort_close only after all workers have stopped using the handle.

## DAC model caveats

DAC encoder/decoder exports can have multiple inputs or outputs and model-
specific dynamic dimensions. This first adapter rejects multi-input and
multi-output graphs. The float32 ABI is a transport format, not a claim that
integer code tensors are audio samples. Validate exact model names, ranks,
dimensions, and types before connecting it to an audio graph. A later version
should expose named tensor descriptors for multi-I/O models.

## Verification boundary

The C++ source requires the external ONNX Runtime SDK and has not been compiled
in this repository environment. The Cmajor declarations have not been
compiled because a working cmaj toolchain is unavailable. This is an
implementation drop, not a claim of a verified native build.
