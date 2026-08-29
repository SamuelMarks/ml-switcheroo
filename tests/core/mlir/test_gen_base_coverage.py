"""Tests for MLIR gen base coverage."""

import typing

import libcst as cst

from ml_switcheroo.core.mlir.cst import AttributeNode, OperationNode
from ml_switcheroo.core.mlir.gen_base import BaseGeneratorMixin


class DummyGen(BaseGeneratorMixin):
  """Dummy generator."""

  def map_op(self, op: OperationNode) -> None:
    """Map operation."""
    pass


def test_get_attr_list() -> None:
  """Docstring."""
  gen = DummyGen()
  op = OperationNode(
    name="test",
    operands=[],
    results=[],
    attributes=[AttributeNode(name="k", value="[v1, v2]", type_annotation="str")],
    regions=[],
  )
  assert gen._get_attr(op, "k") == "[v1, v2]"


def test_create_dotted_name_empty() -> None:
  """Docstring."""
  gen = DummyGen()
  node: typing.Any = gen._create_dotted_name("")
  assert isinstance(node, cst.Name)
  assert node.value == "unknown"
