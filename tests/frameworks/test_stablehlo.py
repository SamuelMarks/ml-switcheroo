"""Test suite for the Stablehlo module."""

from ml_switcheroo.frameworks.stablehlo import StableHloAdapter
from ml_switcheroo.frameworks.base import InitMode
from ml_switcheroo_ir.schema.ghost import SemanticTier


def test_stablehlo_adapter_init():
  """Verifies the behavior of StableHLO adapter initialization."""
  adapter = StableHloAdapter()
  assert adapter.display_name == "StableHLO (MLIR)"
  assert adapter.ui_priority == 95
  assert adapter.inherits_from is None
  assert adapter._mode == InitMode.GHOST


def test_stablehlo_properties():
  """Verifies the behavior of StableHLO properties."""
  adapter = StableHloAdapter()
  assert adapter.import_alias == ("stablehlo", "stablehlo")
  assert adapter.import_namespaces == {}
  assert SemanticTier.ARRAY_API in adapter.supported_tiers
  traits = adapter.structural_traits
  assert traits.module_base is None
  config = adapter.test_config
  assert "// module attributes" in config["import"]
  assert adapter.harness_imports == []
  assert adapter.get_harness_init_code() == ""
  assert adapter.get_to_numpy_code() == "return str(obj)"
  assert adapter.declared_magic_args == []
  assert adapter.rng_seed_methods == []
  defs = adapter.definitions
  assert isinstance(defs, dict)
  specs = adapter.specifications
  assert specs == {}
  snapshot = {}
  adapter.apply_wiring(snapshot)
  assert snapshot == {}
  examples = adapter.get_tiered_examples()
  assert "tier1_math" in examples
  assert "stablehlo.abs" in examples["tier1_math"]
