"""Tests for the auxiliary transformer."""

import typing


import libcst as _cst

from ml_switcheroo.core.rewriter.passes.auxiliary import AuxiliaryTransformer

cst: typing.Any = _cst


class MockSemantics:
  """Test helper class MockSemantics."""

  def get_definition(self, name: typing.Any) -> typing.Any:
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
  """Test helper class DummyContext."""

  def __init__(self) -> None:
    """Initialize dummy context."""
    self.target_fw: typing.Any = "flax_nnx"
    self.semantics: typing.Any = MockSemantics()


def test_auxiliary_leave_decorator_no_name() -> None:
  """Verify auxiliary leave decorator no name behavior."""
  p: typing.Any = AuxiliaryTransformer(context=DummyContext())
  node: typing.Any = cst.parse_module("@foo()()\ndef x(): pass").body[0].decorators[0]

  # We must add an _get_qualified_name method that returns None on the transformer
  # to mock it, or just let it return None naturally for complex calls.
  # Actually _get_qualified_name is a method on AuxiliaryTransformer.
  res: typing.Any = p.leave_Decorator(node, node)
  assert res == node


def test_auxiliary_leave_decorator_fw_not_in_variants() -> None:
  """Verify auxiliary leave decorator fw not in variants behavior."""
  p: typing.Any = AuxiliaryTransformer(context=DummyContext())
  node: typing.Any = cst.parse_module("@my_dec\ndef x(): pass").body[0].decorators[0]

  # Needs a mock _get_qualified_name to return "my_dec"
  p._get_qualified_name = lambda x: "my_dec"

  res: typing.Any = p.leave_Decorator(node, node)
  assert res == node
