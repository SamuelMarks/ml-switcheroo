"""Test module."""

from ml_switcheroo.frameworks.latex_dsl import LatexDSLAdapter
from ml_switcheroo.semantics.schema import PluginTraits


def test_latex_dsl_adapter_get_device_check_syntax():
  """Test function."""
  adapter = LatexDSLAdapter()
  assert adapter.get_device_check_syntax() == "True"


def test_latex_dsl_adapter_definitions():
  """Test function."""
  import ml_switcheroo.frameworks.latex_dsl as ldsl
  from unittest.mock import patch

  with patch("ml_switcheroo.frameworks.latex_dsl.load_definitions", return_value={}):
    adapter = ldsl.LatexDSLAdapter()
    defs = adapter.definitions
    assert "Module" in defs
    assert "Conv2d" in defs
    assert "Linear" in defs


def test_latex_dsl_adapter_properties():
  """Test function."""
  adapter = LatexDSLAdapter()
  assert adapter.get_device_syntax("cpu") == ""
  assert adapter.get_rng_split_syntax("rng", "key") == ""
  assert adapter.get_serialization_imports() == []
  assert adapter.get_serialization_syntax("save", "file") == ""
  assert adapter.get_weight_conversion_imports() == []
  assert adapter.get_weight_load_code("path") == "# Weights not supported in LaTeX mode"
  assert adapter.get_tensor_to_numpy_expr("t") == "t"
  assert adapter.get_weight_save_code("state", "path") == "# Weights not supported in LaTeX mode"
  assert adapter.convert("data") == "data"
  assert isinstance(adapter.plugin_traits, PluginTraits)

  examples = adapter.get_tiered_examples()
  assert "tier1_math" in examples
  assert "tier2_neural" in examples
  assert "tier3_extras" in examples
