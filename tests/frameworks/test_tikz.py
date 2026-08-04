"""Test suite for the Tikz module."""

from ml_switcheroo.frameworks.tikz import TikzAdapter
from ml_switcheroo.frameworks.base import InitMode
from ml_switcheroo_ir.schema.ghost import SemanticTier


def test_tikz_adapter_init():
  """Verifies the behavior of TikZ adapter initialization."""
  adapter = TikzAdapter()
  assert adapter.display_name == "TikZ (LaTeX)"
  assert adapter.ui_priority == 1000
  assert adapter.inherits_from is None
  assert adapter._mode == InitMode.GHOST


def test_tikz_properties():
  """Verifies the behavior of TikZ properties."""
  adapter = TikzAdapter()
  assert adapter.import_alias == ("tikz", "tikz")
  assert adapter.import_namespaces == {}
  assert SemanticTier.NEURAL in adapter.supported_tiers
  traits = adapter.structural_traits
  assert traits.module_base is None
  config = adapter.test_config
  assert "% latex package imports here" in config["import"]
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
  assert "tier2_neural" in examples
  assert "\\begin{tikzpicture}" in examples["tier2_neural"]


def test_tikz_missing_coverage():
  """Verifies untested methods of TikzAdapter."""
  adapter = TikzAdapter()

  # Traits
  assert adapter.plugin_traits is not None

  # Device & RNG
  assert adapter.get_device_syntax("cpu") == ""
  assert adapter.get_device_check_syntax() == "True"
  assert adapter.get_rng_split_syntax("rng", "key") == ""

  # Serialization
  assert adapter.get_serialization_imports() == []
  assert adapter.get_serialization_syntax("save", "path") == ""
  assert adapter.get_weight_conversion_imports() == []
  assert adapter.get_weight_load_code("path") == "# Weights not supported in TikZ mode"
  assert adapter.get_tensor_to_numpy_expr("my_tensor") == "my_tensor"
  assert adapter.get_weight_save_code("state", "path") == "# Weights not supported in TikZ mode"

  # Convert
  assert adapter.convert(123) == "123"
