"""
Tests for the LaTeX DSL Framework Adapter.

Verifies:
1. Registration in `_ADAPTER_REGISTRY`.
2. Protocol compliance.
3. Backend route availability.
4. Semantic Definitions (`midl` namespace) are populated correctly.
"""

from ml_switcheroo.frameworks.latex_dsl import LatexDSLAdapter
from ml_switcheroo.frameworks.base import (
  _ADAPTER_REGISTRY,
  get_adapter,
  InitMode,
  StandardCategory,
)
from ml_switcheroo.core.latex.parser import LatexParser
from ml_switcheroo.enums import SemanticTier
from ml_switcheroo.semantics.schema import StructuralTraits
from ml_switcheroo.compiler.registry import get_backend_class, is_isa_target


def test_registry_integration():
  """Verify adapter is registered."""
  assert "latex_dsl" in _ADAPTER_REGISTRY
  adapter = get_adapter("latex_dsl")
  assert isinstance(adapter, LatexDSLAdapter)
  assert adapter.display_name == "LaTeX DSL (MIDL)"


def test_initialization_defaults():
  """Verify Ghost mode init."""
  adapter = LatexDSLAdapter()
  assert adapter._mode == InitMode.GHOST
  assert adapter.search_modules == []
  assert adapter.import_alias == ("midl", "midl")


def test_backend_registered():
  """
  Verify the backend registry routes 'latex_dsl' correctly.
  Replaces deprecated `create_emitter` test.
  """
  assert is_isa_target("latex_dsl")
  cls = get_backend_class("latex_dsl")
  assert cls is not None
  assert cls.__name__ == "LatexBackend"


def test_parser_factory():
  """Verify parser factory still exists on adapter for Ingestion phase."""
  adapter = LatexDSLAdapter()
  parser = adapter.create_parser(r"\begin{DefModel} \end{DefModel}")
  assert isinstance(parser, LatexParser)


def test_definitions_exist():
  """
  Verify that definitions are populated for crucial ops.
  """
  adapter = LatexDSLAdapter()
  defs = adapter.definitions

  assert "Conv2d" in defs
  conv = defs["Conv2d"]
  assert conv.api == "midl.Conv2d"
  assert conv.args["in_channels"] == "arg_0"

  assert "Linear" in defs
  lin = defs["Linear"]
  assert lin.api == "midl.Linear"


def test_import_namespaces():
  """
  Verify that the 'midl' namespace is declared as Neural Tier.
  """
  adapter = LatexDSLAdapter()
  ns = adapter.import_namespaces
  assert "midl" in ns
  assert ns["midl"].tier == SemanticTier.NEURAL
  assert ns["midl"].recommended_alias == "midl"


def test_example_code_validity():
  """
  Verify the example code is valid LaTeX structure.
  """
  code = LatexDSLAdapter().get_tiered_examples()["tier2_neural"]
  assert r"\begin{DefModel}" in code
  assert r"\Attribute" in code


def test_helper_method_safety():
  """
  Verify protocol methods return safe empty values.
  """
  adapter = LatexDSLAdapter()

  assert adapter.get_device_syntax("cuda") == ""
  assert adapter.get_rng_split_syntax("r", "k") == ""
  assert adapter.get_serialization_syntax("save", "f") == ""
  assert adapter.collect_api(StandardCategory.LAYER) == []

  snap = {}
  adapter.apply_wiring(snap)
  assert snap == {}


def test_structural_traits_for_source():
  """
  Verify traits are defined for when LaTeX is used as SOURCE.
  """
  adapter = LatexDSLAdapter()
  traits = adapter.structural_traits

  assert traits.module_base == "midl.Module"
  assert traits.forward_method == "forward"
