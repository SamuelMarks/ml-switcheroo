"""Test suite for the Stablehlo module."""

import typing

from ml_switcheroo_ir.schema.ghost import SemanticTier

from ml_switcheroo.frameworks.base import InitMode
from ml_switcheroo.frameworks.stablehlo import StableHloAdapter
from ml_switcheroo.semantics.schema import PluginTraits


def test_stablehlo_adapter_init() -> None:
  """Verifies the behavior of StableHLO adapter initialization."""
  adapter = StableHloAdapter()
  assert adapter.display_name == "StableHLO (MLIR)"
  assert adapter.ui_priority == 95
  assert adapter.inherits_from is None
  assert adapter._mode == InitMode.LIVE


def test_stablehlo_properties() -> None:
  """Verifies the behavior of StableHLO properties."""
  adapter = StableHloAdapter()
  assert adapter.import_alias == ("stablehlo", "stablehlo")
  assert adapter.import_namespaces == {}
  assert SemanticTier.ARRAY_API in adapter.supported_tiers
  traits: typing.Any = adapter.structural_traits
  assert traits.module_base is None
  config: dict[str, typing.Any] = adapter.test_config
  assert "import" in config
  assert adapter.harness_imports == []
  assert "xla_bridge" in adapter.get_harness_init_code()
  assert "np.asarray(obj)" in adapter.get_to_numpy_code()
  assert adapter.declared_magic_args == []
  assert adapter.rng_seed_methods == []
  defs: typing.Any = adapter.definitions
  assert isinstance(defs, dict)
  specs: typing.Any = adapter.specifications
  assert specs == {}
  snapshot: dict[str, typing.Any] = {}
  adapter.apply_wiring(snapshot)
  assert snapshot == {}
  examples: dict[str, str] = adapter.get_tiered_examples()
  assert "tier1_math" in examples
  assert "stablehlo.abs" in examples["tier1_math"]


def test_stablehlo_missing_coverage() -> None:
  """Docstring."""
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


# --- Merged from test_stablehlo_extra.py ---


def test_stablehlo_missing_methods() -> None:
  """Docstring."""
  adapter = StableHloAdapter()

  assert adapter.get_device_syntax("cpu") == "// Target: cpu"
  assert adapter.get_device_check_syntax() == "True"
  assert adapter.get_rng_split_syntax("rng", "key") == ""
  assert adapter.get_serialization_imports() == []
  assert adapter.get_serialization_syntax("save", "file", "obj") == ""
  assert adapter.get_serialization_syntax("load", "file") == ""
  assert adapter.get_weight_conversion_imports() == []
  assert adapter.get_weight_load_code("path") == "# Weights not supported in StableHLO mode"
  assert adapter.get_tensor_to_numpy_expr("t") == "t"
  assert adapter.get_weight_save_code("state", "path") == "# Weights not supported in StableHLO mode"

  assert isinstance(adapter.plugin_traits, PluginTraits)
  assert adapter.convert("data") == "data"

  defs: typing.Any = adapter.definitions
  assert isinstance(defs, dict)

  ex: dict[str, str] = adapter.get_tiered_examples()
  assert "tier1_math" in ex
  assert "tier2_neural" in ex
  assert "tier3_extras" in ex

  adapter.apply_wiring({})
  assert adapter.get_doc_url("api") is None
