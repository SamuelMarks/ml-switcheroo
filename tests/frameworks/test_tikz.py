"""
Tests for TikZ Framework Adapter.
"""

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
  code = adapter.get_example_code()

  # Corrected assertions to match the implementation
  assert "\\begin{tikzpicture}" in code
  assert "\\end{tikzpicture}" in code

  tiered = adapter.get_tiered_examples()
  assert "tier2_neural" in tiered
  assert tiered["tier2_neural"] == code
