"""
Tests for Extensible Fuzzer Backends & Metadata Properties.

Verifies that:
1. Adapters are registered correctly.
2. Metadata properties (search_modules, import_alias, inherited_from) are accessible.
3. Conversion logic remains intact.

Note: We assume core frameworks (Torch/JAX) might not be installed in the test env.
Tests mock the module-level variables in the adapters (e.g. `ml_switcheroo.frameworks.torch.torch`)
to simulate presence and force `InitMode.LIVE`.
"""

import sys
import numpy as np
from unittest.mock import MagicMock, patch

from ml_switcheroo.frameworks import (
  register_framework,
  get_adapter,
)
from ml_switcheroo.frameworks.base import _ADAPTER_REGISTRY, InitMode

# Concrete imports for type checking (Directly from submodules)
from ml_switcheroo.frameworks.tensorflow import TensorFlowAdapter
from ml_switcheroo.frameworks.mlx import MLXAdapter
from ml_switcheroo.frameworks.paxml import PaxmlAdapter
from ml_switcheroo.frameworks.numpy import NumpyAdapter
from ml_switcheroo.frameworks.torch import TorchAdapter
from ml_switcheroo.frameworks.jax import JaxAdapter

from ml_switcheroo.testing.fuzzer import InputFuzzer

# --- Metadata Tests ---


def test_torch_metadata():
  """
  Verify Torch adapter metadata. Mocks torch presence to ensure
  search_modules returns the Live list for validation.
  """
  # Simulate torch being present to force InitMode.LIVE
  # Since imports are top-level in the module, patch.dict on sys.modules is insufficient
  # unless we reload. Instead, we patch the module attribute directly.
  with patch("ml_switcheroo.frameworks.torch.torch", MagicMock()):
    # Retrieve a fresh adapter which checks the module variable in __init__
    adapter = TorchAdapter()

    assert adapter.display_name == "PyTorch"
    assert adapter.inherits_from is None
    # Updated: 'torch' root removed from search_modules to prevent recursion issues
    assert "torch.nn" in adapter.search_modules
    assert "torch.linalg" in adapter.search_modules
    assert adapter.import_alias == ("torch", "torch")


def test_jax_metadata():
  """
  Verify JAX adapter metadata. Mocks jax presence.
  """
  # Simulate jax being present to force InitMode.LIVE
  with patch("ml_switcheroo.frameworks.jax.jax", MagicMock()):
    adapter = JaxAdapter()

    assert "JAX" in adapter.display_name
    assert "jax.numpy" in adapter.search_modules
    assert adapter.import_alias == ("jax.numpy", "jnp")


def test_paxml_metadata_inheritance():
  """
  Verify PaxML adapter metadata. Mocks praxis presence.
  """
  # Simulate praxis being present to force InitMode.LIVE
  with patch("ml_switcheroo.frameworks.paxml.praxis", MagicMock()):
    adapter = PaxmlAdapter()

    assert adapter.display_name == "PaxML / Praxis"
    assert adapter.inherits_from == "jax"
    assert "praxis.layers" in adapter.search_modules
    assert adapter.import_alias == ("praxis.layers", "pl")


def test_custom_adapter_properties():
  """Verify that a newly registered adapter can define properties."""

  @register_framework("fastai")
  class FastAIAdapter:
    display_name = "FastAI"
    inherits_from = "torch"
    search_modules = ["fastai.vision"]

    @property
    def import_alias(self):
      return ("fastai", "fastai")

    def convert(self, x):
      return x

  adapter = get_adapter("fastai")
  assert adapter.display_name == "FastAI"
  assert adapter.inherits_from == "torch"
  assert adapter.search_modules == ["fastai.vision"]


# --- Existing Conversion Tests ---


def test_registry_defaults():
  assert "torch" in _ADAPTER_REGISTRY
  assert "jax" in _ADAPTER_REGISTRY
  assert "numpy" in _ADAPTER_REGISTRY
  assert "paxml" in _ADAPTER_REGISTRY
  assert isinstance(get_adapter("torch"), TorchAdapter)


def test_missing_adapter_returns_none():
  assert get_adapter("unknown_fw") is None


def test_torch_converter_simulation():
  # Simulate libraries presence via mocking sys.modules used in convert logic
  mock_torch = MagicMock()
  mock_torch.from_numpy.side_effect = lambda x: f"Torch({x})"

  with patch.dict(sys.modules, {"torch": mock_torch}):
    # Use patch context to ensure 'torch' variable in module is also set
    # if adapter methods reference it directly, though convert usually imports locally.
    # TorchAdapter.convert does local import inside exception block.
    adapter = TorchAdapter()
    val = adapter.convert([1, 2])

    # Input fuzzer usually creates numpy arrays or lists.
    arr = np.array([1, 2])
    val = adapter.convert(arr)
    assert val == f"Torch({arr})"


def test_paxml_converter_installed():
  """Verify PaxML adapter uses JAX numpy conversion logic."""
  mock_jax = MagicMock()
  mock_jnp = MagicMock()
  mock_jnp.array.side_effect = lambda x: f"JNP({x})"

  # Ensure the parent's attribute points to the child submodule mock
  mock_jax.numpy = mock_jnp

  # Patch execution environment imports
  with patch.dict(sys.modules, {"jax": mock_jax, "jax.numpy": mock_jnp}):
    adapter = PaxmlAdapter()
    val = adapter.convert([1, 2])
    assert val == "JNP([1, 2])"


def test_fuzzer_delegates_to_adapter():
  class MockAdapter:
    display_name = "Mock"

    def convert(self, _data):
      return "converted"

  register_framework("mock_fw")(MockAdapter)
  fuzzer = InputFuzzer()
  raw_data = {"x": np.array([1, 2])}
  result = fuzzer.adapt_to_framework(raw_data, "mock_fw")
  assert result["x"] == "converted"
