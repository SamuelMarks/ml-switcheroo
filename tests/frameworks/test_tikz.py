"""
Tests for TikZ Framework Adapter.
"""

from ml_switcheroo.enums import SemanticTier
from ml_switcheroo.frameworks.tikz import TikzAdapter
from ml_switcheroo.frameworks.base import StandardCategory, InitMode, get_adapter, _ADAPTER_REGISTRY


def test_tikz_adapter_registry():
  """Verify TikZ is registered and retrievable."""
  assert "tikz" in _ADAPTER_REGISTRY
  adapter = get_adapter("tikz")
  assert isinstance(adapter, TikzAdapter)


def test_tikz_adapter_initialization():
  """Verify initialization defaults."""
  adapter = TikzAdapter()
  assert adapter.display_name == "TikZ (LaTeX)"
  assert adapter.ui_priority == 1000
  assert adapter._mode == InitMode.GHOST
  assert adapter.search_modules == []
  assert adapter.import_alias == ("tikz", "tikz")


def test_tikz_defaults_return_empty():
  """Verify protocol methods return safe empty defaults."""
  adapter = TikzAdapter()

  assert adapter.get_device_syntax("cuda") == ""
  assert adapter.get_rng_split_syntax("rng", "key") == ""
  assert adapter.collect_api(StandardCategory.LAYER) == []

  snap = {}
  adapter.apply_wiring(snap)
  assert snap == {}


def test_tikz_conversion():
  """Verify data conversion is string identity."""
  adapter = TikzAdapter()
  assert adapter.convert(123) == "123"


def test_tikz_examples():
  """Verify sample code generation."""
  adapter = TikzAdapter()
  examples = adapter.get_tiered_examples()

  assert "tier2_neural" in examples
  code = examples["tier2_neural"]
  # Corrected assertions to match the implementation
  assert "\\begin{tikzpicture}" in code
  assert "\\end{tikzpicture}" in code


def test_tikz_all_methods():
  adapter = TikzAdapter()
  assert adapter.search_modules == []
  assert adapter.unsafe_submodules == set()
  assert adapter.import_namespaces == {}
  assert adapter.discovery_heuristics == {}
  assert "import" in adapter.test_config
  assert adapter.harness_imports == []
  assert adapter.get_harness_init_code() == ""
  assert adapter.get_to_numpy_code() == "return str(obj)"
  assert SemanticTier.NEURAL in adapter.supported_tiers
  assert adapter.declared_magic_args == []
  assert adapter.structural_traits is not None
  assert adapter.plugin_traits is not None
  assert isinstance(adapter.definitions, dict)
  assert adapter.specifications == {}
  assert adapter.rng_seed_methods == []
  assert adapter.get_device_check_syntax() == "True"
  assert adapter.get_serialization_imports() == []
  assert adapter.get_serialization_syntax("op", "f") == ""
  assert adapter.get_weight_conversion_imports() == []
  assert adapter.get_weight_load_code("path") == "# Weights not supported in TikZ mode"
  assert adapter.get_tensor_to_numpy_expr("tensor") == "tensor"
  assert adapter.get_weight_save_code("state", "path") == "# Weights not supported in TikZ mode"
  assert adapter.get_doc_url("any") is None
