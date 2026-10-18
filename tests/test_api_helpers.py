"""Tests for the API helpers mixin used by transformers."""

import typing


import libcst as _cst

from ml_switcheroo.core.rewriter.passes.api_helpers import ApiHelpersMixin

cst: typing.Any = _cst


class DummyContext:
  """Test helper class DummyContext."""

  def __init__(self) -> None:
    """Initialize the dummy context."""
    self.target_fw: typing.Any = "flax_nnx"
    self.alias_map: typing.Any = {"my_root": "canonical_root"}


class DummyTransformer(ApiHelpersMixin):
  """Dummy transformer using the API helpers mixin."""

  def __init__(self) -> None:
    """Initialize the dummy transformer."""
    self.context: typing.Any = DummyContext()

  def _cst_to_string(self, node: typing.Any) -> typing.Any:
    """Convert a CST node to a string.

    Args:
        node (cst.CSTNode): The node.

    Returns:
        str: String value.
    """
    return getattr(node, "value", str(node))


def test_api_helpers_get_qualified_name_short() -> None:
  """Verify api helpers get qualified name short behavior."""
  p: typing.Any = DummyTransformer()
  node: typing.Any = cst.Name("my_root")
  res: typing.Any = p._get_qualified_name(node)
  assert res == "canonical_root"


def test_api_helpers_inject_stmts_to_body_simple() -> None:
  """Verify api helpers inject stmts to body simple behavior."""
  p: typing.Any = DummyTransformer()
  node: typing.Any = cst.parse_statement("def foo(): pass")
  new_stmts: typing.Any = [cst.parse_statement("x = 1").body[0]]
  res: typing.Any = p._inject_stmts_to_body(node, new_stmts)
  assert isinstance(res.body, cst.IndentedBlock)
  assert len(res.body.body) == 2


def test_api_helpers_inject_argument_to_signature_default_comma() -> None:
  """Verify api helpers inject argument to signature default comma behavior."""
  p: typing.Any = DummyTransformer()
  node: typing.Any = cst.parse_statement("def foo(self): pass")
  # Provide a dummy method to simulate create_dotted_name since it creates AST nodes
  setattr(p, "_create_dotted_name", lambda name: cst.Name(name))
  res: typing.Any = p._inject_argument_to_signature(node, "new_arg", None)

  params: typing.Any = list(res.params.params)
  assert len(params) == 2
  assert params[0].name.value == "self"
  assert isinstance(params[0].comma, cst.Comma)
  assert params[1].name.value == "new_arg"
