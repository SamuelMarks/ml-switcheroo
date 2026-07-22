"""Test suite for the Rdna module."""

from ml_switcheroo.frameworks.rdna import RdnaAdapter
from ml_switcheroo.frameworks.base import InitMode
from ml_switcheroo_ir.schema.ghost import SemanticTier


def test_rdna_adapter_init():
  """Verifies the behavior of RDNA adapter initialization."""
  adapter = RdnaAdapter()
  assert adapter.display_name == "AMD RDNA"
  assert adapter.ui_priority == 151
  assert adapter.inherits_from is None
  assert adapter._mode == InitMode.GHOST
  assert adapter.target_arch == "gfx1030"


def test_rdna_properties():
  """Verifies the behavior of RDNA properties."""
  adapter = RdnaAdapter()
  assert adapter.import_alias == ("rdna", "asm")
  assert adapter.import_namespaces == {}
  assert SemanticTier.ARRAY_API in adapter.supported_tiers
  traits = adapter.structural_traits
  assert traits.module_base is None
  config = adapter.test_config
  assert "; RDNA Header" in config["import"]
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
  assert "v_add_f32" in examples["tier1_math"]
