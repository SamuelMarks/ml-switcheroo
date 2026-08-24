"""Test module."""

from ml_switcheroo.core.mlir.types import IntegerType, FloatType, ComplexType, TensorType, FunctionType


def test_complex_type():
  """Test element."""
  f32 = FloatType("f32")
  c = ComplexType(f32)
  assert c.to_string() == "complex<f32>"


def test_tensor_type():
  """Test element."""
  f32 = FloatType("f32")
  t0 = TensorType(f32, None)
  assert t0.to_string() == "tensor<*xf32>"

  t1 = TensorType(f32, [])
  assert t1.to_string() == "tensor<f32>"

  t2 = TensorType(f32, [2, "?", 4])
  assert t2.to_string() == "tensor<2x?x4xf32>"


def test_function_type():
  """Test element."""
  f32 = FloatType("f32")
  i32 = IntegerType(32)

  f1 = FunctionType([f32, i32], [])
  assert f1.to_string() == "(f32, i32) -> ()"

  f2 = FunctionType([f32], [f32, i32])
  assert f2.to_string() == "(f32) -> (f32, i32)"

  f3 = FunctionType([f32], [f32])
  assert f3.to_string() == "(f32) -> f32"
