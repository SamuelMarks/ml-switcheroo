"""
Tests for TikZ Framework Adapter.
"""

from ml_switcheroo.frameworks.tikz import TikzAdapter
from ml_switcheroo.frameworks.base import StandardCategory, InitMode, get_adapter, _ADAPTER_REGISTRY


def test_tikz_adapter_registry():
  """Verify TikZ is registered and retrievable."""
  # Ensure it's in the registry (populated by import in __init__ or direct import in test)
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
  assert adapter.get_device_check_syntax() == "False"
  assert adapter.get_rng_split_syntax("rng", "key") == ""
  assert adapter.get_serialization_imports() == []
  assert adapter.get_serialization_syntax("save", "f") == ""
  assert adapter.collect_api(StandardCategory.LAYER) == []

  # Wiring should be no-op
  snap = {}
  adapter.apply_wiring(snap)
  assert snap == {}


def test_tikz_dictionaries_empty():
  """Verify mappings/traits are empty."""
  adapter = TikzAdapter()
  assert adapter.definitions == {}
  assert adapter.specifications == {}
  assert adapter.import_namespaces == {}
  assert adapter.discovery_heuristics == {}
  assert adapter.test_config == {}


def test_tikz_conversion():
  """Verify data conversion is string identity."""
  adapter = TikzAdapter()
  assert adapter.convert(123) == "123"
  assert adapter.convert([1, 2]) == "[1, 2]"


def test_tikz_examples():
  """Verify sample code generation."""
  adapter = TikzAdapter()
  code = adapter.get_example_code()
  assert "\\begin{tikzpicture}" in code
  assert "\\node" in code

  tiered = adapter.get_tiered_examples()
  assert "tier2_neural" in tiered
  assert tiered["tier2_neural"] == code
