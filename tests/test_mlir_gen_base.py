"""Test module."""

import libcst as cst
from ml_switcheroo.core.mlir.cst import OperationNode, AttributeNode
from ml_switcheroo.core.mlir.gen_base import BaseGeneratorMixin


class TestGenerator(BaseGeneratorMixin):
  """Test element."""

  pass


def test_get_attr():
  """Test element."""
  gen = TestGenerator()
  op = OperationNode(name="test.op")
  assert gen._get_attr(op, "missing") is None

  op = OperationNode(name="test.op", attributes=[AttributeNode(name="foo", value="bar")])
  assert gen._get_attr(op, "foo") == "bar"

  op = OperationNode(name="test.op", attributes=[AttributeNode(name="arr", value=["1", "2"])])
  assert gen._get_attr(op, "arr") == "[1, 2]"


def test_create_dotted_name():
  """Test element."""
  gen = TestGenerator()

  node = gen._create_dotted_name("")
  assert isinstance(node, cst.Name)
  assert node.value == "unknown"

  node = gen._create_dotted_name("torch")
  assert isinstance(node, cst.Name)
  assert node.value == "torch"

  node = gen._create_dotted_name("torch.nn.functional")
  assert isinstance(node, cst.Attribute)
  assert node.attr.value == "functional"
  assert isinstance(node.value, cst.Attribute)
  assert node.value.attr.value == "nn"
  assert isinstance(node.value.value, cst.Name)
  assert node.value.value.value == "torch"
