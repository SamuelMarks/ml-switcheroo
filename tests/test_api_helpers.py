"""Tests for the API helpers mixin used by transformers."""

from typing import Dict, List

import libcst as cst

from ml_switcheroo.core.rewriter.passes.api_helpers import ApiHelpersMixin


class DummyContext:
  """Docstring."""

  def __init__(self) -> None:
    """Initialize the dummy context."""
    self.target_fw: str = "flax_nnx"
    self.alias_map: Dict[str, str] = {"my_root": "canonical_root"}


class DummyTransformer(ApiHelpersMixin):
  """Dummy transformer using the API helpers mixin."""

  def __init__(self) -> None:
    """Initialize the dummy transformer."""
    self.context: DummyContext = DummyContext()  # type: ignore

  def _cst_to_string(self, node: cst.CSTNode) -> str:
    """Convert a CST node to a string.

    Args:
        node (cst.CSTNode): The node.

    Returns:
        str: String value.
    """
    return getattr(node, "value", str(node))


def test_api_helpers_get_qualified_name_short() -> None:
  """Docstring."""
  p: DummyTransformer = DummyTransformer()
  node: cst.Name = cst.Name("my_root")
  res: str = p._get_qualified_name(node)
  assert res == "canonical_root"


def test_api_helpers_inject_stmts_to_body_simple() -> None:
  """Docstring."""
  p: DummyTransformer = DummyTransformer()
  node: cst.FunctionDef = cst.parse_statement("def foo(): pass")  # type: ignore
  new_stmts: List[cst.BaseStatement] = [cst.parse_statement("x = 1").body[0]]  # type: ignore
  res: cst.FunctionDef = p._inject_stmts_to_body(node, new_stmts)
  assert isinstance(res.body, cst.IndentedBlock)
  assert len(res.body.body) == 2


def test_api_helpers_inject_argument_to_signature_default_comma() -> None:
  """Docstring."""
  p: DummyTransformer = DummyTransformer()
  node: cst.FunctionDef = cst.parse_statement("def foo(self): pass")  # type: ignore
  # Provide a dummy method to simulate create_dotted_name since it creates AST nodes
  p._create_dotted_name = lambda name: cst.Name(name)  # type: ignore
  res: cst.FunctionDef = p._inject_argument_to_signature(node, "new_arg", None)

  params: List[cst.Param] = list(res.params.params)
  assert len(params) == 2
  assert params[0].name.value == "self"
  assert isinstance(params[0].comma, cst.Comma)
  assert params[1].name.value == "new_arg"
