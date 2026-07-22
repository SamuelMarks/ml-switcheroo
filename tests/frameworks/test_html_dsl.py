"""Test suite for the Html Dsl module."""

from unittest.mock import patch
from ml_switcheroo.frameworks.html_dsl import HtmlDSLAdapter
from ml_switcheroo.frameworks.base import InitMode, StandardMap
from ml_switcheroo_ir.schema.ghost import SemanticTier


def test_html_dsl_init():
  """Verifies the behavior of HTML DSL initialization."""
  adapter = HtmlDSLAdapter()
  assert adapter.display_name == "HTML Grid DSL"
  assert adapter.ui_priority == 980
  assert adapter.inherits_from is None
  assert adapter._mode == InitMode.GHOST


def test_html_dsl_create_parser():
  """Verifies the behavior of HTML DSL create parser."""
  adapter = HtmlDSLAdapter()
  parser = adapter.create_parser("<code>hello</code>")
  assert parser is not None
  assert getattr(parser, "source", None) == "<code>hello</code>"


def test_html_dsl_properties():
  """Verifies the behavior of HTML DSL properties."""
  adapter = HtmlDSLAdapter()
  assert adapter.import_alias == ("html_dsl", "dsl")
  assert "html_dsl" in adapter.import_namespaces
  assert SemanticTier.NEURAL in adapter.supported_tiers
  traits = adapter.structural_traits
  assert traits.module_base == "html_dsl.Module"
  assert traits.forward_method == "forward"
  assert traits.init_method_name == "__init__"
  assert traits.requires_super_init is True
  assert adapter.test_config == {}
  assert adapter.harness_imports == []
  assert adapter.get_harness_init_code() == ""
  assert adapter.get_to_numpy_code() == "return str(obj)"
  assert adapter.declared_magic_args == []
  assert adapter.rng_seed_methods == []
  defs = adapter.definitions
  assert "Module" in defs
  assert defs["Module"].api == "html_dsl.Module"
  assert adapter.specifications == {}
  snapshot = {}
  adapter.apply_wiring(snapshot)
  assert snapshot == {}
  assert adapter.get_doc_url("html_dsl.Module") is None
  examples = adapter.get_tiered_examples()
  assert "tier2_neural" in examples
  assert 'class="grid"' in examples["tier2_neural"]


@patch("ml_switcheroo.frameworks.html_dsl.load_definitions")
def test_html_dsl_definitions_already_present(mock_load):
  """Verifies the behavior of HTML DSL definitions already present."""
  mock_load.return_value = {"Module": StandardMap(api="existing.Module"), "Conv2d": StandardMap(api="existing.Conv2d")}
  adapter = HtmlDSLAdapter()
  defs = adapter.definitions
  assert defs["Module"].api == "existing.Module"
  assert defs["Conv2d"].api == "existing.Conv2d"
