"""Tests for the MLIR strict type system."""

from ml_switcheroo.core.mlir.types import (
  IntegerType,
  FloatType,
  ComplexType,
  TensorType,
  FunctionType,
)


def test_integer_type() -> None:
  """Test IntegerType string formatting."""
  assert IntegerType(32).to_string() == "i32"
  assert IntegerType(1).to_string() == "i1"


def test_float_type() -> None:
  """Test FloatType string formatting."""
  assert FloatType("f32").to_string() == "f32"
  assert FloatType("bf16").to_string() == "bf16"


def test_complex_type() -> None:
  """Test ComplexType string formatting."""
  assert ComplexType(FloatType("f64")).to_string() == "complex<f64>"


def test_tensor_type() -> None:
  """Test TensorType string formatting for various shapes."""
  f32 = FloatType("f32")
  # Unranked
  assert TensorType(f32, None).to_string() == "tensor<*xf32>"
  # Scalar (0D)
  assert TensorType(f32, []).to_string() == "tensor<f32>"
  # Ranked static
  assert TensorType(f32, [10, 20]).to_string() == "tensor<10x20xf32>"
  # Ranked dynamic
  assert TensorType(f32, [10, "?"]).to_string() == "tensor<10x?xf32>"


def test_function_type() -> None:
  """Test FunctionType string formatting."""
  f32 = FloatType("f32")
  t_f32 = TensorType(f32, None)
  t_i32 = TensorType(IntegerType(32), [10])

  # No inputs, no results
  assert FunctionType([], []).to_string() == "() -> ()"
  # One input, one result
  assert FunctionType([t_f32], [t_f32]).to_string() == "(tensor<*xf32>) -> tensor<*xf32>"
  # Multiple inputs, multiple results
  assert (
    FunctionType([t_f32, t_i32], [t_i32, t_f32]).to_string()
    == "(tensor<*xf32>, tensor<10xi32>) -> (tensor<10xi32>, tensor<*xf32>)"
  )
