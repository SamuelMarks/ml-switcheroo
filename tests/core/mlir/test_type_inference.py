"""Tests for the MLIR static type inference engine."""

import libcst as cst

from ml_switcheroo.core.mlir.type_inference import TypeInferencePass, parse_py_type_to_mlir
from ml_switcheroo.core.mlir.types import FloatType, IntegerType, TensorType


def test_parse_py_type_to_mlir() -> None:
  """Docstring."""
  assert parse_py_type_to_mlir("int") == IntegerType(32)
  assert parse_py_type_to_mlir("float") == FloatType("f32")
  assert parse_py_type_to_mlir("bool") == IntegerType(1)
  assert parse_py_type_to_mlir("Tensor") == TensorType(FloatType("f32"), None)
  assert parse_py_type_to_mlir("unknown") == FloatType("!sw.unknown")


def test_type_inference_pass() -> None:
  """Docstring."""
  code = """
def my_func(a):
    b = 5.0
    c = 10
    d = a
    return d
"""
  module = cst.parse_module(code)

  # Initialize pass with type for 'a'
  initial_env = {"a": TensorType(FloatType("f32"), [10, 20])}
  infer_pass = TypeInferencePass(initial_env=initial_env)

  module.visit(infer_pass)

  assert infer_pass.env["b"] == FloatType("f32")
  assert infer_pass.env["c"] == IntegerType(32)
  assert infer_pass.env["d"] == TensorType(FloatType("f32"), [10, 20])

  assert len(infer_pass.return_types) == 1
  assert infer_pass.return_types[0] == TensorType(FloatType("f32"), [10, 20])


def test_type_inference_pass_empty_return() -> None:
  """Docstring."""
  code = """
def my_func():
    return
"""
  module = cst.parse_module(code)
  infer_pass = TypeInferencePass()
  module.visit(infer_pass)
  assert len(infer_pass.return_types) == 0


def test_type_inference_pass_unknown_expression() -> None:
  """Docstring."""
  code = """
def my_func():
    b = unknown_func()
    return b
"""
  module = cst.parse_module(code)
  infer_pass = TypeInferencePass()
  module.visit(infer_pass)
  assert infer_pass.env["b"] == TensorType(FloatType("f32"), None)
  assert infer_pass.return_types[0] == TensorType(FloatType("f32"), None)
