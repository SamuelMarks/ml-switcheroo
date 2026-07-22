"""Test suite for the Latex Dsl module."""

from unittest.mock import patch
from ml_switcheroo.frameworks.latex_dsl import LatexDSLAdapter
from ml_switcheroo.frameworks.base import InitMode, StandardMap
from ml_switcheroo_ir.schema.ghost import SemanticTier


def test_latex_dsl_init():
  """Verifies the behavior of LaTeX DSL initialization."""
  adapter = LatexDSLAdapter()
  assert adapter.display_name == "LaTeX DSL (MIDL)"
  assert adapter.ui_priority == 98
  assert adapter.inherits_from is None
  assert adapter._mode == InitMode.GHOST


def test_latex_dsl_create_parser():
  """Verifies the behavior of LaTeX DSL create parser."""
  adapter = LatexDSLAdapter()
  parser = adapter.create_parser("y = |x|")
  assert parser is not None
  assert getattr(parser, "source", None) == "y = |x|"


def test_latex_dsl_properties():
  """Verifies the behavior of LaTeX DSL properties."""
  adapter = LatexDSLAdapter()
  assert adapter.import_alias == ("midl", "midl")
  assert "midl" in adapter.import_namespaces
  assert SemanticTier.NEURAL in adapter.supported_tiers
  traits = adapter.structural_traits
  assert traits.module_base == "midl.Module"
  config = adapter.test_config
  assert "% latex package imports" in config["import"]
  assert adapter.harness_imports == []
  assert adapter.get_harness_init_code() == ""
  assert adapter.get_to_numpy_code() == "return str(obj)"
  assert adapter.declared_magic_args == []
  assert adapter.rng_seed_methods == []
  defs = adapter.definitions
  assert "Module" in defs
  specs = adapter.specifications
  assert "Conv2d" in specs
  snapshot = {}
  adapter.apply_wiring(snapshot)
  assert snapshot == {}
  assert adapter.get_doc_url("anything") is None
  examples = adapter.get_tiered_examples()
  assert "tier1_math" in examples


@patch("ml_switcheroo.frameworks.latex_dsl.load_definitions")
def test_latex_dsl_definitions_already_present(mock_load):
  """Verifies the behavior of LaTeX DSL definitions already present."""
  mock_load.return_value = {
    "Module": StandardMap(api="existing.Module"),
    "Conv2d": StandardMap(api="existing.Conv2d"),
    "Linear": StandardMap(api="existing.Linear"),
  }
  adapter = LatexDSLAdapter()
  defs = adapter.definitions
  assert defs["Module"].api == "existing.Module"
