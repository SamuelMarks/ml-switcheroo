"""
Tests for the get_to_numpy_code protocol implementation.

Verifies that all registered adapters implement `get_to_numpy_code`
and return valid Python strings for data conversion.
"""

import pytest
from ml_switcheroo.frameworks.base import get_adapter
from ml_switcheroo.frameworks import available_frameworks
import ml_switcheroo.frameworks.torch
import ml_switcheroo.frameworks.jax
import ml_switcheroo.frameworks.tensorflow
import ml_switcheroo.frameworks.keras
import ml_switcheroo.frameworks.numpy
import ml_switcheroo.frameworks.mlx


def test_torch_implementation():
  adapter = ml_switcheroo.frameworks.torch.TorchAdapter()
  code = adapter.get_to_numpy_code()
  assert "detach" in code
  assert "cpu().numpy()" in code


def test_jax_implementation():
  adapter = ml_switcheroo.frameworks.jax.JaxCoreAdapter()
  code = adapter.get_to_numpy_code()
  assert "__array__" in code
  assert "np.array" in code


def test_tensorflow_implementation():
  adapter = ml_switcheroo.frameworks.tensorflow.TensorFlowAdapter()
  code = adapter.get_to_numpy_code()
  assert "numpy()" in code
  assert "hasattr(obj, 'numpy')" in code


def test_keras_implementation():
  adapter = ml_switcheroo.frameworks.keras.KerasAdapter()
  code = adapter.get_to_numpy_code()
  assert "numpy()" in code


def test_numpy_implementation():
  adapter = ml_switcheroo.frameworks.numpy.NumpyAdapter()
  code = adapter.get_to_numpy_code()
  assert "isinstance(obj, np.ndarray)" in code


def test_mlx_implementation():
  adapter = ml_switcheroo.frameworks.mlx.MLXAdapter()
  code = adapter.get_to_numpy_code()
  # MLX uses tolist fallback or array conversion
  assert "tolist" in code


def test_all_adapters_comply(isolate_framework_registry):
  """
  Iterates all registered frameworks to ensure protocol compliance.
  """
  fws = available_frameworks()
  for fw in fws:
    adapter = get_adapter(fw)
    if not adapter:
      continue

    try:
      code = adapter.get_to_numpy_code()
      assert isinstance(code, str), f"{fw}: get_to_numpy_code must return str"
    except AttributeError:
      pytest.fail(f"{fw} adapter does not implement get_to_numpy_code")
