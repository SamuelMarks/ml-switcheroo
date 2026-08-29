"""Tests for the auxiliary transformer."""

from typing import Any, Dict, Optional, Tuple, Union

import libcst as cst

from ml_switcheroo.core.rewriter.passes.auxiliary import AuxiliaryTransformer


class MockSemantics:
  """Docstring."""

  def get_definition(self, name: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Retrieve a mock definition based on name.

    Args:
        name (str): Definition name.

    Returns:
        Optional[Tuple[str, Dict[str, Any]]]: Definition tuple.
    """
    if name == "my_dec":
      return "id", {"variants": {"other_fw": {}}}
    return None


class DummyContext:
  """Docstring."""

  def __init__(self) -> None:
    """Initialize dummy context."""
    self.target_fw: str = "flax_nnx"
    self.semantics: MockSemantics = MockSemantics()


def test_auxiliary_leave_decorator_no_name() -> None:
  """Docstring."""
  p: AuxiliaryTransformer = AuxiliaryTransformer(context=DummyContext())  # type: ignore
  node: cst.Decorator = cst.parse_module("@foo()()\ndef x(): pass").body[0].decorators[0]  # type: ignore

  # We must add an _get_qualified_name method that returns None on the transformer
  # to mock it, or just let it return None naturally for complex calls.
  # Actually _get_qualified_name is a method on AuxiliaryTransformer.
  res: Union[cst.CSTNode, cst.Decorator, cst.RemovalSentinel] = p.leave_Decorator(node, node)
  assert res == node


def test_auxiliary_leave_decorator_fw_not_in_variants() -> None:
  """Docstring."""
  p: AuxiliaryTransformer = AuxiliaryTransformer(context=DummyContext())  # type: ignore
  node: cst.Decorator = cst.parse_module("@my_dec\ndef x(): pass").body[0].decorators[0]  # type: ignore

  # Needs a mock _get_qualified_name to return "my_dec"
  p._get_qualified_name = lambda x: "my_dec"  # type: ignore

  res: Union[cst.CSTNode, cst.Decorator, cst.RemovalSentinel] = p.leave_Decorator(node, node)
  assert res == node
