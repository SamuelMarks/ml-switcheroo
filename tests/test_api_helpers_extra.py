"""Tests for the API helpers mixin used by transformers."""

import libcst as cst
from ml_switcheroo.core.rewriter.passes.api_helpers import ApiHelpersMixin


class DummyContext:
  """Dummy context for testing."""

  def __init__(self):
    """Initialize the dummy context."""
    self.target_fw = "flax_nnx"
    self.alias_map = {"my_root": "canonical_root"}


class DummyTransformer(ApiHelpersMixin):
  """Dummy transformer using the API helpers mixin."""

  def __init__(self):
    """Initialize the dummy transformer."""
    self.context = DummyContext()

  def _cst_to_string(self, node):
    """Convert a CST node to a string."""
    return node.value


def test_api_helpers_get_qualified_name_short():
  """Test getting a short qualified name from an alias."""
  p = DummyTransformer()
  node = cst.Name("my_root")
  res = p._get_qualified_name(node)
  assert res == "canonical_root"


def test_api_helpers_inject_stmts_to_body_simple():
  """Test injecting statements into a simple block body."""
  p = DummyTransformer()
  node = cst.parse_statement("def foo(): pass")
  new_stmts = [cst.parse_statement("x = 1").body[0]]
  res = p._inject_stmts_to_body(node, new_stmts)
  assert isinstance(res.body, cst.IndentedBlock)
  assert len(res.body.body) == 2


def test_api_helpers_inject_argument_to_signature_default_comma():
  """Test injecting an argument with a trailing comma to a signature."""
  p = DummyTransformer()
  node = cst.parse_statement("def foo(self): pass")
  # Provide a dummy method to simulate create_dotted_name since it creates AST nodes
  p._create_dotted_name = lambda name: cst.Name(name)
  res = p._inject_argument_to_signature(node, "new_arg", None)

  params = res.params.params
  assert len(params) == 2
  assert params[0].name.value == "self"
  assert isinstance(params[0].comma, cst.Comma)
  assert params[1].name.value == "new_arg"
