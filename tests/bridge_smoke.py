"""Native C ABI smoke test using a generated one-input/one-output ONNX graph."""
from __future__ import annotations
import ctypes
import os
from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

def main() -> None:
    library = ctypes.CDLL(os.environ["RQ_BRIDGE_LIBRARY"])
    library.rq_ort_open.argtypes = [ctypes.c_char_p, ctypes.c_int32]
    library.rq_ort_open.restype = ctypes.c_int64
    library.rq_ort_run_f32.argtypes = [
        ctypes.c_int64, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int32, ctypes.POINTER(ctypes.c_float), ctypes.c_int64,
        ctypes.POINTER(ctypes.c_int64), ctypes.c_int32, ctypes.POINTER(ctypes.c_int32),
    ]
    library.rq_ort_run_f32.restype = ctypes.c_int64
    library.rq_ort_close.argtypes = [ctypes.c_int64]
    model_path = Path(os.environ["RUNNER_TEMP"]) / "bridge-smoke.onnx"
    weights = np.arange(12, dtype=np.float32).reshape(4, 3) / 10
    x_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, 4])
    y_info = helper.make_tensor_value_info("y", TensorProto.FLOAT, [None, 3])
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["x", "w"], ["y"])],
        "bridge_smoke", [x_info], [y_info],
        [numpy_helper.from_array(weights, name="w")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, model_path)
    handle = library.rq_ort_open(os.fsencode(model_path), 1)
    assert handle > 0, "rq_ort_open failed"
    try:
        source = np.array([[0.5, -1.0, 2.0, 0.25]], dtype=np.float32)
        expected = source @ weights
        input_values = (ctypes.c_float * source.size)(*source.ravel())
        input_dims = (ctypes.c_int64 * 2)(1, 4)
        output_values = (ctypes.c_float * 3)()
        output_dims = (ctypes.c_int64 * 4)()
        output_rank = ctypes.c_int32(-1)
        count = library.rq_ort_run_f32(
            handle, input_values, input_dims, 2, output_values, 3,
            output_dims, 4, ctypes.byref(output_rank),
        )
        assert count == 3
        assert output_rank.value == 2
        assert list(output_dims[:2]) == [1, 3]
        np.testing.assert_allclose(np.ctypeslib.as_array(output_values), expected.ravel(), rtol=1e-6)
    finally:
        library.rq_ort_close(handle)

if __name__ == "__main__":
    main()
