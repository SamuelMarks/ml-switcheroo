"""Tests for MLIR gen base coverage."""

import libcst as cst
from ml_switcheroo.core.mlir.gen_base import BaseGeneratorMixin
from ml_switcheroo.core.mlir.nodes import OperationNode, AttributeNode


class DummyGen(BaseGeneratorMixin):
  """Dummy generator."""

  def map_op(self, op):
    """Map operation."""
    pass


def test_get_attr_list():
  """Test get attr list."""
  gen = DummyGen()
  op = OperationNode(
    name="test", operands=[], results=[], attributes=[AttributeNode("k", ["v1", "v2"], "str")], regions=[]
  )
  assert gen._get_attr(op, "k") == "[v1, v2]"


def test_create_dotted_name_empty():
  """Test create dotted name empty."""
  gen = DummyGen()
  node = gen._create_dotted_name("")
  assert isinstance(node, cst.Name)
  assert node.value == "unknown"
