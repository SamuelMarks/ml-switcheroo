"""Tests for the auxiliary transformer."""

import libcst as cst
from ml_switcheroo.core.rewriter.passes.auxiliary import AuxiliaryTransformer


class MockSemantics:
  """Mock semantics implementation for tests."""

  def get_definition(self, name):
    """Retrieve a mock definition based on name."""
    if name == "my_dec":
      return "id", {"variants": {"other_fw": {}}}
    return None


class DummyContext:
  """Dummy context for auxiliary tests."""

  def __init__(self):
    """Initialize dummy context."""
    self.target_fw = "flax_nnx"
    self.semantics = MockSemantics()


def test_auxiliary_leave_decorator_no_name():
  """Test leaving a decorator when no qualified name is found."""
  p = AuxiliaryTransformer(context=DummyContext())
  node = cst.parse_module("@foo()()\ndef x(): pass").body[0].decorators[0]

  # We must add an _get_qualified_name method that returns None on the transformer
  # to mock it, or just let it return None naturally for complex calls.
  # Actually _get_qualified_name is a method on AuxiliaryTransformer.
  res = p.leave_Decorator(node, node)
  assert res == node


def test_auxiliary_leave_decorator_fw_not_in_variants():
  """Test leaving a decorator when the target framework is not in the semantics variants."""
  p = AuxiliaryTransformer(context=DummyContext())
  node = cst.parse_module("@my_dec\ndef x(): pass").body[0].decorators[0]

  # Needs a mock _get_qualified_name to return "my_dec"
  p._get_qualified_name = lambda x: "my_dec"

  res = p.leave_Decorator(node, node)
  assert res == node
