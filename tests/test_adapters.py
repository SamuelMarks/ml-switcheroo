"""
Tests for Extensible Fuzzer Backends (Feature 028).
"""

import sys
import numpy as np
from unittest.mock import MagicMock, patch

from ml_switcheroo.testing.adapters import (
  register_adapter,
  get_adapter,
  TensorFlowAdapter,
  MLXAdapter,
  NumpyAdapter,
  _ADAPTER_REGISTRY,
)
from ml_switcheroo.testing.fuzzer import InputFuzzer


def test_registry_defaults():
  assert "torch" in _ADAPTER_REGISTRY
  assert "jax" in _ADAPTER_REGISTRY
  assert "numpy" in _ADAPTER_REGISTRY
  assert "tensorflow" in _ADAPTER_REGISTRY
  assert "mlx" in _ADAPTER_REGISTRY
  assert isinstance(get_adapter("tensorflow"), TensorFlowAdapter)
  assert isinstance(get_adapter("mlx"), MLXAdapter)


def test_custom_adapter_registration():
  class ORTAdapter:
    def convert(self, data):
      return f"ort_tensor({data})"

  register_adapter("onnxruntime", ORTAdapter)
  adapter = get_adapter("onnxruntime")
  assert adapter is not None
  assert adapter.convert(5) == "ort_tensor(5)"


def test_missing_adapter_returns_none():
  assert get_adapter("unknown_fw") is None


def test_tensorflow_converter_installed():
  mock_tf = MagicMock()
  mock_tf.convert_to_tensor.side_effect = lambda x: f"TF({x})"
  with patch.dict(sys.modules, {"tensorflow": mock_tf}):
    adapter = TensorFlowAdapter()
    val = adapter.convert(10)
    assert val == "TF(10)"


def test_tensorflow_converter_missing():
  with patch.dict(sys.modules, {"tensorflow": None}):
    with patch("builtins.__import__", side_effect=ImportError):
      adapter = TensorFlowAdapter()
      val = adapter.convert(10)
      assert val == 10


def test_mlx_converter_installed():
  mock_mlx = MagicMock()
  mock_mx = MagicMock()
  mock_mx.array.side_effect = lambda x: f"MX({x})"
  mock_mlx.core = mock_mx
  with patch.dict(sys.modules, {"mlx": mock_mlx, "mlx.core": mock_mx}):
    adapter = MLXAdapter()
    val = adapter.convert([1, 2])
    assert val == "MX([1, 2])"


def test_mlx_converter_missing():
  with patch.dict(sys.modules, {"mlx.core": None}):
    with patch("builtins.__import__", side_effect=ImportError):
      adapter = MLXAdapter()
      val = adapter.convert([1, 2])
      assert val == [1, 2]


def test_numpy_adapter_recursion():
  """
  Verify NumpyAdapter recursively normalizes lists/tuples.
  """
  adapter = NumpyAdapter()

  # Mock a tensor-like obj
  mock_tensor = MagicMock()
  mock_tensor.detach.return_value.cpu.return_value.numpy.return_value = np.array(100)

  data = [1, mock_tensor, 3]
  res = adapter.convert(data)

  assert isinstance(res, list)
  assert res[0] == 1
  assert res[1] == np.array(100)
  assert res[2] == 3


def test_fuzzer_delegates_to_adapter():
  class MockAdapter:
    def convert(self, _data):
      return "converted"

  register_adapter("mock_fw", MockAdapter)
  fuzzer = InputFuzzer()
  raw_data = {"x": np.array([1, 2])}
  result = fuzzer.adapt_to_framework(raw_data, "mock_fw")
  assert result["x"] == "converted"


def test_fuzzer_fallback_passthrough():
  fuzzer = InputFuzzer()
  raw_data = {"x": np.array([1, 2])}
  result = fuzzer.adapt_to_framework(raw_data, "weird_fw")
  assert result["x"] is raw_data["x"]
