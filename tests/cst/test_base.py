"""Unit tests for the Core Concrete Syntax Tree (CST) framework.

This module verifies the behavior of foundational CST structures, including trivia,
nodes, post-initialization logic, visitation, and transformation mechanisms.
"""

import pytest
from ml_switcheroo.core.cst.base import Trivia, CSTNode, CSTVisitor, CSTTransformer
from dataclasses import dataclass, field
from typing import List


@dataclass
class DummyNode(CSTNode):
  """A concrete mockup of a CSTNode used for testing basic CST behaviors.

  Attributes:
      name (str): The name identifier for this node. Defaults to "dummy".
  """

  name: str = "dummy"

  def to_text(self) -> str:
    """Renders the dummy node and its associated trivia to its string representation.

    Returns:
        str: The fully reconstituted source text including leading trivia, node name,
             and trailing trivia.
    """
    return f"{''.join(t.text for t in self.leading_trivia)}{self.name}{''.join(t.text for t in self.trailing_trivia)}"


@dataclass
class ContainerNode(CSTNode):
  """A mock container node that holds a child node and a list of sibling nodes.

  This helper class enables testing parent-child relationships, visitation,
  and recursive tree transformation processes.

  Attributes:
      child (DummyNode): A single child node of type DummyNode.
      children (List[DummyNode]): A list of nested DummyNode instances.
  """

  child: DummyNode = field(default_factory=DummyNode)
  children: List[DummyNode] = field(default_factory=list)

  def to_text(self) -> str:
    """Renders the container node back to a string representation.

    Returns:
        str: An empty string as the default container behavior for testing purposes.
    """
    return ""


def test_trivia() -> None:
  """Verifies that the Trivia object correctly retains its non-semantic text content.

  This test instantiates a Trivia block with whitespace and asserts that its
  literal representation matches the input.
  """
  t = Trivia("  ")
  assert t.text == "  "


def test_cstnode_post_init() -> None:
  """Verifies that CSTNode's __post_init__ properly processes and sanitizes trivia.

  Specifically, this tests that string arguments for trivia are automatically
  coerced into lists of Trivia objects, and that None is gracefully handled and
  coerced into empty lists.
  """
  node = DummyNode(leading_trivia=" ", trailing_trivia=" ")
  assert len(node.leading_trivia) == 1
  assert node.leading_trivia[0].text == " "
  assert len(node.trailing_trivia) == 1
  assert node.trailing_trivia[0].text == " "

  node2 = DummyNode(leading_trivia=None, trailing_trivia=None)
  assert node2.leading_trivia == []
  assert node2.trailing_trivia == []


def test_cstnode_to_text() -> None:
  """Tests string representation and abstract base behaviors of CSTNode.

  This verifies that calling `to_text` or `str` on a subclass successfully
  combines trivia with internal properties, and that calling them on the
  base CSTNode class directly raises a NotImplementedError.
  """
  node = DummyNode(leading_trivia=" ", trailing_trivia=" ")
  assert node.to_text() == " dummy "
  assert str(node) == " dummy "

  node2 = CSTNode()
  with pytest.raises(NotImplementedError):
    node2.to_text()

  with pytest.raises(NotImplementedError):
    str(node2)


def test_cstvisitor() -> None:
  """Verifies the traversal capabilities of the CSTVisitor pattern.

  The test sets up a tree structure with nested nodes and ensures that visiting
  the parent correctly triggers visitor callbacks on all child and sibling nodes.
  """
  visited = []

  class MyVisitor(CSTVisitor):
    """A custom visitor implementation that tracks visited dummy nodes.

    This class overrides specific node visitor methods to accumulate node names
    during the CST traversal.
    """

    def visit_DummyNode(self, node: DummyNode) -> None:
      """Visitor callback specifically triggered when visiting a DummyNode.

      Appends the visited node's name to the outer context's list and continues
      the traversal of child nodes.

      Args:
          node (DummyNode): The DummyNode currently being visited.
      """
      visited.append(node.name)
      self.generic_visit(node)

  v = MyVisitor()
  child1 = DummyNode(name="child1")
  child2 = DummyNode(name="child2")
  container = ContainerNode(child=child1, children=[child2])

  v.visit(container)
  assert visited == ["child1", "child2"]


def test_csttransformer() -> None:
  """Verifies the mutation and tree-rewrite capabilities of the CSTTransformer.

  This test defines a custom transformer that alters node properties, runs it
  against a mock tree structure, and asserts that nodes are properly transformed
  while non-node members remain unchanged.
  """

  class MyTransformer(CSTTransformer):
    """A custom transformer implementation that modifies DummyNode names.

    This class targets DummyNode instances in the CST and appends a suffix
    to their name attributes during traversal.
    """

    def transform_DummyNode(self, node: DummyNode) -> DummyNode:
      """Transforms a DummyNode by appending a designated suffix to its name.

      This method mutates the DummyNode's name in-place, then recursively
      transforms any child nodes before returning the updated node.

      Args:
          node (DummyNode): The DummyNode to be transformed.

      Returns:
          DummyNode: The transformed DummyNode.
      """
      node.name += "_transformed"
      return self.generic_transform(node)

  child1 = DummyNode(name="child1")
  child2 = DummyNode(name="child2")
  container = ContainerNode(child=child1, children=[child2, "not_a_node"])

  t = MyTransformer()
  new_container = t.transform(container)

  assert new_container.child.name == "child1_transformed"
  assert new_container.children[0].name == "child2_transformed"
  assert new_container.children[1] == "not_a_node"


def test_csttransformer_no_specific() -> None:
  """Verifies CSTTransformer behavior when no node-specific transform method exists.

  This ensures that when a transformer does not define a custom hook for a given
  node type, it falls back to the generic transformation mechanism, preserving
  the existing tree nodes unmodified.
  """

  class EmptyTransformer(CSTTransformer):
    """An empty CSTTransformer subclass that defines no custom transformation rules.

    Used to test the default fallback execution paths in tree transformations.
    """

    pass

  child1 = DummyNode(name="child1")
  container = ContainerNode(child=child1)

  t = EmptyTransformer()
  new_container = t.transform(container)

  assert new_container.child.name == "child1"
