"""Test suite for the Protocol Numpy Code module."""

import pytest

import ml_switcheroo.frameworks.jax
import ml_switcheroo.frameworks.keras
import ml_switcheroo.frameworks.mlx
import ml_switcheroo.frameworks.numpy
import ml_switcheroo.frameworks.tensorflow
import ml_switcheroo.frameworks.torch
from ml_switcheroo.frameworks import available_frameworks
from ml_switcheroo.frameworks.base import get_adapter


def test_torch_implementation() -> None:
  """Verify `TorchAdapter` emits `detach` for torch implementation."""
  adapter = ml_switcheroo.frameworks.torch.TorchAdapter()
  code: str = adapter.get_to_numpy_code()
  assert "detach" in code
  assert "cpu().numpy()" in code


def test_jax_implementation() -> None:
  """Verify `JaxCoreAdapter` emits `np.asarray` for JAX implementation."""
  adapter = ml_switcheroo.frameworks.jax.JaxCoreAdapter()
  code: str = adapter.get_to_numpy_code()
  assert "np.asarray" in code


def test_tensorflow_implementation() -> None:
  """Verify `TensorFlowAdapter` emits `numpy()` for tensorflow implementation."""
  adapter = ml_switcheroo.frameworks.tensorflow.TensorFlowAdapter()
  code: str = adapter.get_to_numpy_code()
  assert "numpy()" in code
  assert "AttributeError" in code


def test_keras_implementation() -> None:
  """Verify `KerasAdapter` emits `numpy()` for keras implementation."""
  adapter = ml_switcheroo.frameworks.keras.KerasAdapter()
  code: str = adapter.get_to_numpy_code()
  assert "numpy()" in code


def test_numpy_implementation() -> None:
  """Verify `NumpyAdapter` emits `isinstance(obj, np.ndarray)` for numpy implementation."""
  adapter = ml_switcheroo.frameworks.numpy.NumpyAdapter()
  code: str = adapter.get_to_numpy_code()
  assert "isinstance(obj, np.ndarray)" in code


def test_mlx_implementation() -> None:
  """Verify `MLXAdapter` emits `tolist` for MLX implementation."""
  adapter = ml_switcheroo.frameworks.mlx.MLXAdapter()
  code: str = adapter.get_to_numpy_code()
  assert "tolist" in code


def test_all_adapters_comply(isolate_framework_registry: None) -> None:
  """Verify `available_frameworks` handles all adapters comply as expected.

  Args:
      isolate_framework_registry: Input `isolate_framework_registry` parameter value (`None`).
  """
  fws: list[str] = available_frameworks()
  assert "torch" in fws
  for fw in fws:
    adapter = get_adapter(fw)
    if not adapter:
      continue
    try:
      code: str = adapter.get_to_numpy_code()
      assert isinstance(code, str), f"{fw}: get_to_numpy_code must return str"
      assert len(code.strip()) >= 1, f"{fw}: get_to_numpy_code must be non-empty"
    except AttributeError:
      pytest.fail(f"{fw} adapter does not implement get_to_numpy_code")
