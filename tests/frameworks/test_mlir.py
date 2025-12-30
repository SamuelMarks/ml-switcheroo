"""
Tests for MLIR Framework Adapter.
"""

from ml_switcheroo.frameworks.base import _ADAPTER_REGISTRY, get_adapter
from ml_switcheroo.frameworks.mlir import MlirAdapter


def test_mlir_adapter_registration():
  """Verify registration."""
  assert "mlir" in _ADAPTER_REGISTRY
  adapter = get_adapter("mlir")
  assert isinstance(adapter, MlirAdapter)
  assert adapter.display_name == "MLIR (Intermediate)"


def test_mlir_properties_defaults():
  """Verify empty/default properties."""
  adapter = MlirAdapter()
  assert adapter.search_modules == []
  assert adapter.import_alias == ("mlir", "sw")
  assert adapter.definitions == {}
  assert adapter.specifications == {}
  assert adapter.test_config == {}
  assert adapter.unsafe_submodules == set()
  assert adapter.get_device_syntax("cuda") == ""
  assert adapter.convert(123) == "123"


def test_mlir_example_code():
  """Verify example code generation."""
  assert "MLIR" in MlirAdapter.get_example_code()
  adapter = MlirAdapter()
  examples = adapter.get_tiered_examples()
  assert "tier1_math" in examples
  assert "tier2_neural" in examples
