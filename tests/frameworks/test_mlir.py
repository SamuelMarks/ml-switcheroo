"""Test suite for the Mlir module."""

import typing

from ml_switcheroo_ir.schema.ghost import SemanticTier

from ml_switcheroo.frameworks.base import InitMode
from ml_switcheroo.frameworks.mlir import MlirAdapter
from ml_switcheroo.semantics.schema import PluginTraits


def test_mlir_adapter_init() -> None:
  """Verifies the behavior of MLIR adapter initialization."""
  adapter = MlirAdapter()
  assert adapter.display_name == "MLIR (Intermediate)"
  assert adapter.ui_priority == 90
  assert adapter.inherits_from is None
  assert adapter._mode == InitMode.GHOST


def test_mlir_properties() -> None:
  """Verifies the behavior of MLIR properties."""
  adapter = MlirAdapter()
  assert adapter.import_alias == ("mlir", "sw")
  assert adapter.import_namespaces == {}
  assert SemanticTier.ARRAY_API in adapter.supported_tiers
  assert SemanticTier.NEURAL in adapter.supported_tiers
  traits: typing.Any = adapter.structural_traits
  assert traits.module_base is None
  config: dict[str, typing.Any] = adapter.test_config
  assert "// module attributes" in config["import"]
  assert adapter.harness_imports == []
  assert adapter.get_harness_init_code() == ""
  assert adapter.get_to_numpy_code() == "return str(obj)"
  assert adapter.declared_magic_args == []
  assert adapter.rng_seed_methods == []
  defs: typing.Any = adapter.definitions
  assert isinstance(defs, dict)
  specs: typing.Any = adapter.specifications
  assert specs == {}
  snapshot: dict[str, typing.Any] = {}
  adapter.apply_wiring(snapshot)
  assert snapshot == {}
  assert adapter.get_doc_url("anything") is None
  examples: dict[str, str] = adapter.get_tiered_examples()
  assert "tier1_math" in examples
  assert "sw.module" in examples["tier1_math"]
  assert "sw.module" in adapter.get_example_code()


# --- Merged from test_mlir_extra.py ---


def test_mlir_adapter_missing_methods() -> None:
  """Docstring."""
  adapter = MlirAdapter()

  assert adapter.import_namespaces == {}

  test_cfg: dict[str, typing.Any] = adapter.test_config
  assert "import" in test_cfg

  assert adapter.harness_imports == []
  assert adapter.get_harness_init_code() == ""
  assert adapter.get_to_numpy_code() == "return str(obj)"
  assert adapter.declared_magic_args == []

  traits: typing.Any = adapter.structural_traits
  assert traits.module_base is None

  assert isinstance(adapter.plugin_traits, PluginTraits)
  assert adapter.specifications == {}
  assert adapter.rng_seed_methods == []
  assert adapter.get_device_syntax("cpu") == "// Target: cpu"
  assert adapter.get_device_check_syntax() == "True"
  assert adapter.get_rng_split_syntax("rng", "key") == "// Split RNG: rng -> key"
  assert adapter.get_serialization_imports() == []
  assert adapter.get_serialization_syntax("save", "file", "obj") == "// Save obj to file"
  assert adapter.get_serialization_syntax("load", "file") == "// Load from file"

  assert adapter.get_weight_conversion_imports() == []
  assert adapter.get_weight_load_code("path") == "# Weights loading not supported in MLIR adapter"
  assert adapter.get_tensor_to_numpy_expr("t") == "t"
  assert adapter.get_weight_save_code("state", "path") == "# Weights saving not supported in MLIR adapter"

  adapter.apply_wiring({})
  assert adapter.get_doc_url("api") is None
  assert adapter.convert("data") == "data"

  assert "tier1_math" in adapter.get_tiered_examples()
  assert "sw.module" in adapter.get_example_code()
