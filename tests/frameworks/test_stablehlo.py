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
  assert adapter._mode == InitMode.LIVE


def test_stablehlo_properties():
  """Verifies the behavior of StableHLO properties."""
  adapter = StableHloAdapter()
  assert adapter.import_alias == ("stablehlo", "stablehlo")
  assert adapter.import_namespaces == {}
  assert SemanticTier.ARRAY_API in adapter.supported_tiers
  traits = adapter.structural_traits
  assert traits.module_base is None
  config = adapter.test_config
  assert "import" in config
  assert adapter.harness_imports == []
  assert "xla_bridge" in adapter.get_harness_init_code()
  assert "np.asarray(obj)" in adapter.get_to_numpy_code()
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


def test_stablehlo_missing_coverage():
  """Verifies untested methods of StableHloAdapter."""
  adapter = StableHloAdapter()

  # Traits
  assert adapter.plugin_traits is not None

  # Device & RNG
  assert adapter.get_device_syntax("cpu") == "// Target: cpu"
  assert adapter.get_device_check_syntax() == "True"
  assert adapter.get_rng_split_syntax("rng", "key") == ""

  # Serialization
  assert adapter.get_serialization_imports() == []
  assert adapter.get_serialization_syntax("load", "path") == ""
  assert adapter.get_weight_conversion_imports() == []
  assert adapter.get_weight_load_code("path") == "# Weights not supported in StableHLO mode"
  assert adapter.get_tensor_to_numpy_expr("my_tensor") == "my_tensor"
  assert adapter.get_weight_save_code("state", "path") == "# Weights not supported in StableHLO mode"

  # Documentation
  assert adapter.get_doc_url("stablehlo.abs") == "https://github.com/openxla/stablehlo/blob/main/docs/spec.md#abs"
  assert adapter.get_doc_url("other") is None

  # Convert
  assert adapter.convert(456) == "456"
