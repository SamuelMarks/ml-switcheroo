"""Tests for the base CST implementation."""

import pytest
from dataclasses import dataclass, field
from typing import List, Any

from ml_switcheroo.core.cst.base import CSTNode, CSTVisitor, CSTTransformer, Trivia


@dataclass
class DummyNode(CSTNode):
  """A simple node for testing."""

  name: str = ""
  child: "CSTNode" = None  # type: ignore
  children: List["CSTNode"] = field(default_factory=list)
  mixed_list: List[Any] = field(default_factory=list)

  def to_text(self) -> str:
    """Returns the text representation."""
    res = "".join(t.text for t in self.leading_trivia)
    res += self.name
    res += "".join(t.text for t in self.trailing_trivia)
    return res


class MockVisitor(CSTVisitor):
  """A test visitor to track visited nodes."""

  def __init__(self) -> None:
    """Initialize the visitor."""
    self.visited_names: List[str] = []

  def visit_DummyNode(self, node: DummyNode) -> None:
    """Visits a DummyNode."""
    self.visited_names.append(node.name)
    self.generic_visit(node)


class MockTransformer(CSTTransformer):
  """A test transformer that renames DummyNodes."""

  def transform_DummyNode(self, node: DummyNode) -> DummyNode:
    """Transforms a DummyNode."""
    node.name = node.name.upper()
    self.generic_transform(node)
    return node


def test_trivia() -> None:
  """Test trivia creation and access."""
  t = Trivia("  ")
  assert t.text == "  "


def test_cstnode_base() -> None:
  """Test that CSTNode base class raises NotImplementedError."""
  node = CSTNode()
  with pytest.raises(NotImplementedError):
    node.to_text()


def test_visitor() -> None:
  """Test the CSTVisitor traversal logic."""
  root = DummyNode(name="root")
  child1 = DummyNode(name="child1")
  child2 = DummyNode(name="child2")
  grandchild = DummyNode(name="grandchild")

  child2.child = grandchild
  root.children = [child1, child2]
  root.mixed_list = ["string", child1, 123]

  visitor = MockVisitor()
  visitor.visit(root)

  assert visitor.visited_names == ["root", "child1", "child2", "grandchild", "child1"]


def test_transformer() -> None:
  """Test the CSTTransformer transformation logic."""
  root = DummyNode(name="root")
  child1 = DummyNode(name="child1")
  child2 = DummyNode(name="child2")
  grandchild = DummyNode(name="grandchild")

  child2.child = grandchild
  root.children = [child1, child2]
  root.mixed_list = ["string", child1, 123]

  transformer = MockTransformer()
  transformed_root = transformer.transform(root)

  assert isinstance(transformed_root, DummyNode)
  assert transformed_root.name == "ROOT"
  assert transformed_root.children[0].name == "CHILD1"  # type: ignore
  assert transformed_root.children[1].name == "CHILD2"  # type: ignore
  assert getattr(transformed_root.children[1], "child").name == "GRANDCHILD"
  assert transformed_root.mixed_list[0] == "string"
  assert transformed_root.mixed_list[1].name == "CHILD1"
  assert transformed_root.mixed_list[2] == 123


def test_cst_no_native_extensions() -> None:
  """Ensure no native parsing extensions are mistakenly imported in cst."""
  import sys

  for module_name in sys.modules:
    assert not module_name.startswith("llvmlite"), "Native extension llvmlite found"
    assert not module_name.startswith("mlir.ir"), "Native extension mlir.ir found"
