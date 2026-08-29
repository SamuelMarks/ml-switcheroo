"""Test suite for the Latex Dsl module."""

import typing
from unittest.mock import MagicMock, patch

from ml_switcheroo_ir.schema.ghost import SemanticTier

from ml_switcheroo.frameworks.base import InitMode, StandardMap
from ml_switcheroo.frameworks.latex_dsl import LatexDSLAdapter
from ml_switcheroo.semantics.schema import PluginTraits


def test_latex_dsl_init() -> None:
  """Verifies the behavior of LaTeX DSL initialization."""
  adapter = LatexDSLAdapter()
  assert adapter.display_name == "LaTeX DSL (MIDL)"
  assert adapter.ui_priority == 98
  assert adapter.inherits_from is None
  assert adapter._mode == InitMode.GHOST


def test_latex_dsl_create_parser() -> None:
  """Verifies the behavior of LaTeX DSL create parser."""
  adapter = LatexDSLAdapter()
  parser: typing.Any = adapter.create_parser("y = |x|")
  assert parser is not None
  assert getattr(parser, "source", None) == "y = |x|"


def test_latex_dsl_properties() -> None:
  """Verifies the behavior of LaTeX DSL properties."""
  adapter = LatexDSLAdapter()
  assert adapter.import_alias == ("midl", "midl")
  assert "midl" in adapter.import_namespaces
  assert SemanticTier.NEURAL in adapter.supported_tiers
  traits: typing.Any = adapter.structural_traits
  assert traits.module_base == "midl.Module"
  config: dict[str, typing.Any] = adapter.test_config
  assert "% latex package imports" in config["import"]
  assert adapter.harness_imports == []
  assert adapter.get_harness_init_code() == ""
  assert adapter.get_to_numpy_code() == "return str(obj)"
  assert adapter.declared_magic_args == []
  assert adapter.rng_seed_methods == []
  defs: typing.Any = adapter.definitions
  assert "Module" in defs
  specs: typing.Any = adapter.specifications
  assert "Conv2d" in specs
  snapshot: dict[str, typing.Any] = {}
  adapter.apply_wiring(snapshot)
  assert snapshot == {}
  assert adapter.get_doc_url("anything") is None
  examples: dict[str, str] = adapter.get_tiered_examples()
  assert "tier1_math" in examples


@patch("ml_switcheroo.frameworks.latex_dsl.load_definitions")
def test_latex_dsl_definitions_already_present(mock_load: MagicMock) -> None:
  """Verifies the behavior of LaTeX DSL definitions already present."""
  mock_load.return_value = {
    "Module": StandardMap(api="existing.Module"),
    "Conv2d": StandardMap(api="existing.Conv2d"),
    "Linear": StandardMap(api="existing.Linear"),
  }
  adapter = LatexDSLAdapter()
  defs: typing.Any = adapter.definitions
  assert defs["Module"].api == "existing.Module"


# --- Merged from test_latex_dsl_extra2.py ---


def test_latex_dsl_adapter_get_device_check_syntax() -> None:
  """Docstring."""
  adapter = LatexDSLAdapter()
  assert adapter.get_device_check_syntax() == "True"


def test_latex_dsl_adapter_definitions() -> None:
  """Docstring."""
  from unittest.mock import patch

  import ml_switcheroo.frameworks.latex_dsl as ldsl

  with patch("ml_switcheroo.frameworks.latex_dsl.load_definitions", return_value={}):
    adapter = ldsl.LatexDSLAdapter()
    defs: typing.Any = adapter.definitions
    assert "Module" in defs
    assert "Conv2d" in defs
    assert "Linear" in defs


def test_latex_dsl_adapter_properties() -> None:
  """Docstring."""
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

  examples: dict[str, str] = adapter.get_tiered_examples()
  assert "tier1_math" in examples
  assert "tier2_neural" in examples
  assert "tier3_extras" in examples


# --- Merged from test_latex_dsl_extra.py ---


def test_latex_dsl_adapter_missing_methods() -> None:
  """Docstring."""
  adapter = LatexDSLAdapter()
  assert adapter.get_device_syntax("cpu") == ""
  assert adapter.get_device_check_syntax() == "True"
  assert adapter.get_rng_split_syntax("rng", "key") == ""
  assert adapter.get_serialization_imports() == []
  assert adapter.get_serialization_syntax("save", "file") == ""
  assert adapter.get_weight_conversion_imports() == []
  assert adapter.get_weight_load_code("path") == "# Weights not supported in LaTeX mode"
  assert adapter.get_tensor_to_numpy_expr("t") == "t"
  assert adapter.get_weight_save_code("state", "path") == "# Weights not supported in LaTeX mode"
  assert adapter.convert("data") == "data"

  assert isinstance(adapter.plugin_traits, PluginTraits)
