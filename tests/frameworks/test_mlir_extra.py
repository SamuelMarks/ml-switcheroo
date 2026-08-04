"""Test module."""

from ml_switcheroo.frameworks.mlir import MlirAdapter
from ml_switcheroo.semantics.schema import PluginTraits


def test_mlir_adapter_missing_methods():
  """Test function."""
  adapter = MlirAdapter()

  assert adapter.import_namespaces == {}

  test_cfg = adapter.test_config
  assert "import" in test_cfg

  assert adapter.harness_imports == []
  assert adapter.get_harness_init_code() == ""
  assert adapter.get_to_numpy_code() == "return str(obj)"
  assert adapter.declared_magic_args == []

  traits = adapter.structural_traits
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
