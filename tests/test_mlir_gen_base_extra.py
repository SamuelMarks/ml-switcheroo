"""Test module."""

from ml_switcheroo.core.mlir.cst import OperationNode, AttributeNode
from ml_switcheroo.core.mlir.gen_base import BaseGeneratorMixin


class TestGenerator(BaseGeneratorMixin):
  """Test element."""

  pass


def test_get_attr_continue():
  """Test element."""
  gen = TestGenerator()
  op = OperationNode(
    name="test.op", attributes=[AttributeNode(name="other", value="val"), AttributeNode(name="foo", value="bar")]
  )
  assert gen._get_attr(op, "foo") == "bar"
