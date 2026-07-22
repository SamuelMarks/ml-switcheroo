"""Test suite for the Mlir module."""

from ml_switcheroo.frameworks.mlir import MlirAdapter
from ml_switcheroo.frameworks.base import InitMode
from ml_switcheroo_ir.schema.ghost import SemanticTier


def test_mlir_adapter_init():
  """Verifies the behavior of MLIR adapter initialization."""
  adapter = MlirAdapter()
  assert adapter.display_name == "MLIR (Intermediate)"
  assert adapter.ui_priority == 90
  assert adapter.inherits_from is None
  assert adapter._mode == InitMode.GHOST


def test_mlir_properties():
  """Verifies the behavior of MLIR properties."""
  adapter = MlirAdapter()
  assert adapter.import_alias == ("mlir", "sw")
  assert adapter.import_namespaces == {}
  assert SemanticTier.ARRAY_API in adapter.supported_tiers
  assert SemanticTier.NEURAL in adapter.supported_tiers
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
  assert adapter.get_doc_url("anything") is None
  examples = adapter.get_tiered_examples()
  assert "tier1_math" in examples
  assert "sw.module" in examples["tier1_math"]
  assert "sw.module" in adapter.get_example_code()
