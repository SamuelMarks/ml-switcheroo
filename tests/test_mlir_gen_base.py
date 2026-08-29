"""Test module."""

import libcst as cst

from ml_switcheroo.core.mlir.cst import AttributeNode, OperationNode
from ml_switcheroo.core.mlir.gen_base import BaseGeneratorMixin


class TestGenerator(BaseGeneratorMixin):
  """Docstring."""

  pass


def test_get_attr() -> None:
  """Docstring."""
  gen: TestGenerator = TestGenerator()
  op: OperationNode = OperationNode(name="test.op")
  assert gen._get_attr(op, "missing") is None

  op2: OperationNode = OperationNode(name="test.op", attributes=[AttributeNode(name="foo", value="bar")])
  assert gen._get_attr(op2, "foo") == "bar"

  op3: OperationNode = OperationNode(name="test.op", attributes=[AttributeNode(name="arr", value=["1", "2"])])
  assert gen._get_attr(op3, "arr") == "[1, 2]"


def test_create_dotted_name() -> None:
  """Docstring."""
  gen: TestGenerator = TestGenerator()

  node: cst.BaseExpression = gen._create_dotted_name("")
  assert isinstance(node, cst.Name)
  assert node.value == "unknown"

  node2: cst.BaseExpression = gen._create_dotted_name("torch")
  assert isinstance(node2, cst.Name)
  assert node2.value == "torch"

  node3: cst.BaseExpression = gen._create_dotted_name("torch.nn.functional")
  assert isinstance(node3, cst.Attribute)
  assert node3.attr.value == "functional"
  assert isinstance(node3.value, cst.Attribute)
  assert node3.value.attr.value == "nn"
  assert isinstance(node3.value.value, cst.Name)
  assert node3.value.value.value == "torch"


# --- Merged from test_mlir_gen_base_extra.py ---


class TestGeneratorExtra(BaseGeneratorMixin):
  """Docstring."""

  pass


def test_get_attr_continue() -> None:
  """Docstring."""
  gen: TestGenerator = TestGenerator()
  op: OperationNode = OperationNode(
    name="test.op", attributes=[AttributeNode(name="other", value="val"), AttributeNode(name="foo", value="bar")]
  )
  assert gen._get_attr(op, "foo") == "bar"
