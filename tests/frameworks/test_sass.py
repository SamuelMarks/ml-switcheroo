"""Test suite for the Sass module."""

from ml_switcheroo.frameworks.sass import SassAdapter
from ml_switcheroo.frameworks.base import InitMode
from ml_switcheroo_ir.schema.ghost import SemanticTier


def test_sass_adapter_init():
  """Verifies the behavior of SASS adapter initialization."""
  adapter = SassAdapter()
  assert adapter.display_name == "NVIDIA SASS"
  assert adapter.ui_priority == 150
  assert adapter.inherits_from is None
  assert adapter._mode == InitMode.GHOST


def test_sass_properties():
  """Verifies the behavior of SASS properties."""
  adapter = SassAdapter()
  assert adapter.import_alias == ("sass", "asm")
  assert adapter.import_namespaces == {}
  assert SemanticTier.ARRAY_API in adapter.supported_tiers
  traits = adapter.structural_traits
  assert traits.module_base is None
  config = adapter.test_config
  assert "// SASS Header" in config["import"]
  assert adapter.harness_imports == []
  assert adapter.get_harness_init_code() == ""
  assert adapter.get_to_numpy_code() == "return str(obj)"
  assert adapter.declared_magic_args == []
  assert adapter.rng_seed_methods == []
  defs = adapter.definitions
  assert isinstance(defs, dict)
  assert defs["Conv2d"].api == "Macro.Conv2d"
  specs = adapter.specifications
  assert specs == {}
  snapshot = {}
  adapter.apply_wiring(snapshot)
  assert snapshot == {}
  examples = adapter.get_tiered_examples()
  assert "tier1_math" in examples
  assert "FADD" in examples["tier1_math"]
