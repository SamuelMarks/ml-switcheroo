"""Test suite for the Protocol Numpy Code module."""

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
  """Verifies the behavior of PyTorch implementation."""
  adapter = ml_switcheroo.frameworks.torch.TorchAdapter()
  code = adapter.get_to_numpy_code()
  assert "detach" in code
  assert "cpu().numpy()" in code


def test_jax_implementation():
  """Verifies the behavior of JAX implementation."""
  adapter = ml_switcheroo.frameworks.jax.JaxCoreAdapter()
  code = adapter.get_to_numpy_code()
  assert "__array__" in code
  assert "np.array" in code


def test_tensorflow_implementation():
  """Verifies the behavior of TensorFlow implementation."""
  adapter = ml_switcheroo.frameworks.tensorflow.TensorFlowAdapter()
  code = adapter.get_to_numpy_code()
  assert "numpy()" in code
  assert "hasattr(obj, 'numpy')" in code


def test_keras_implementation():
  """Verifies the behavior of Keras implementation."""
  adapter = ml_switcheroo.frameworks.keras.KerasAdapter()
  code = adapter.get_to_numpy_code()
  assert "numpy()" in code


def test_numpy_implementation():
  """Verifies the behavior of NumPy implementation."""
  adapter = ml_switcheroo.frameworks.numpy.NumpyAdapter()
  code = adapter.get_to_numpy_code()
  assert "isinstance(obj, np.ndarray)" in code


def test_mlx_implementation():
  """Verifies the behavior of MLX implementation."""
  adapter = ml_switcheroo.frameworks.mlx.MLXAdapter()
  code = adapter.get_to_numpy_code()
  assert "tolist" in code


def test_all_adapters_comply(isolate_framework_registry):
  """Verifies the behavior of all adapters comply."""
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
