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
  """
  Verify properties.
  Updated to expect the actual test_config layout defined in the adapter.
  """
  adapter = MlirAdapter()
  assert adapter.search_modules == []
  assert adapter.import_alias == ("mlir", "sw")

  # Verify strict specification default
  assert adapter.specifications == {}

  # Verify Config is populated with MLIR comment syntax
  config = adapter.test_config
  assert config["import"].startswith("//")
  assert "{np_var}" in config["convert_input"]

  assert adapter.unsafe_submodules == set()

  # Wrapper check
  assert adapter.convert(123) == "123"


def test_mlir_example_code():
  """Verify example code generation."""
  code = MlirAdapter.get_example_code()
  # Check for dialect specific tokens
  assert "sw.module" in code
  assert "sw.func" in code
  assert "sw.op" in code
  # Attributes are rendered with spaces around '=' in MLIR emitter
  # e.g. type = "torch.abs"
  assert 'type = "torch.abs"' in code

  adapter = MlirAdapter()
  examples = adapter.get_tiered_examples()
  assert "tier1_math" in examples
  assert "tier2_neural" in examples
  assert "//" in examples["tier3_extras"]
