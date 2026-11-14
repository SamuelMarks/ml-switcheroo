"""Unit tests for the MLIR static type inference pass (`ml_switcheroo.core.mlir.type_inference`)."""

import libcst as cst

from ml_switcheroo.core.mlir.type_inference import TypeInferencePass, parse_py_type_to_mlir
from ml_switcheroo.core.mlir.types import FloatType, IntegerType, MLIRType, TensorType

_NL = chr(10)


def test_parse_py_type_to_mlir() -> None:
  """Verify `parse_py_type_to_mlir` maps Python type strings to MLIR `BaseType` nodes."""
  assert parse_py_type_to_mlir("int") == IntegerType(32)
  assert parse_py_type_to_mlir("int").to_string() == "i32"
  assert parse_py_type_to_mlir("float") == FloatType("f32")
  assert parse_py_type_to_mlir("float").to_string() == "f32"
  assert parse_py_type_to_mlir("bool") == IntegerType(1)
  assert parse_py_type_to_mlir("bool").to_string() == "i1"
  assert parse_py_type_to_mlir("Tensor") == TensorType(FloatType("f32"), None)
  assert parse_py_type_to_mlir("tensor").to_string() == "tensor<*xf32>"
  assert parse_py_type_to_mlir("array").to_string() == "tensor<*xf32>"
  assert parse_py_type_to_mlir("unknown") == FloatType("!sw.unknown")
  assert parse_py_type_to_mlir("unknown").to_string() == "!sw.unknown"


def test_type_inference_pass_assignments_and_returns() -> None:
  """Verify `TypeInferencePass` propagates literal, variable, call, and return types."""
  code = _NL.join(
    [
      "def my_func(a):",
      "    b = 5.0",
      "    c = 10",
      "    d = a",
      "    obj.attr = 1",
      "    return d",
    ]
  )
  initial_env: dict[str, MLIRType] = {"a": TensorType(FloatType("f32"), [10, 20])}
  infer_pass = TypeInferencePass(initial_env=initial_env)
  cst.parse_module(code).visit(infer_pass)

  assert infer_pass.env["b"] == FloatType("f32")
  assert infer_pass.env["c"] == IntegerType(32)
  assert infer_pass.env["d"] == TensorType(FloatType("f32"), [10, 20])
  assert "obj" not in infer_pass.env
  assert infer_pass.return_types == [TensorType(FloatType("f32"), [10, 20])]


def test_type_inference_pass_empty_and_branching_returns() -> None:
  """Verify empty returns, unknown expressions, conditional returns, and tuple returns."""
  empty_ret = TypeInferencePass()
  cst.parse_module(_NL.join(["def f():", "    return"])).visit(empty_ret)
  assert empty_ret.return_types == []

  unk_expr = TypeInferencePass()
  cst.parse_module(_NL.join(["def f():", "    b = unknown_func()", "    return b"])).visit(unk_expr)
  assert unk_expr.env["b"] == TensorType(FloatType("f32"), None)
  assert unk_expr.return_types == [TensorType(FloatType("f32"), None)]

  branch_ret = TypeInferencePass()
  cst.parse_module(_NL.join(["def f(x):", "    if x:", "        return 1", "    return (1, 2)"])).visit(branch_ret)
  assert len(branch_ret.return_types) == 2
  assert branch_ret.return_types[0] == IntegerType(32)
