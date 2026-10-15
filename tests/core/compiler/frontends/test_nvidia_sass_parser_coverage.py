"""Test suite for nvidia sass parser coverage."""

from lark import Tree

from ml_switcheroo.core.compiler.frontends.nvidia_sass.parser import _get_trivia


def test_get_trivia_empty_children():
  """Verify get trivia empty children behavior."""

  class DummyNode:
    """Test helper class DummyNode."""

    def __init__(self):
      """Initialize test helper instance."""
      self.children = []

  assert _get_trivia(DummyNode()) == []


def test_get_trivia_no_children():
  """Verify get trivia no children behavior."""

  class DummyNode:
    """Test helper class DummyNode."""

    pass

  assert _get_trivia(DummyNode()) == []


def test_get_trivia_from_children():
  """Verify get trivia from children behavior."""

  class DummyNodeWithTrivia:
    """Test helper class DummyNodeWithTrivia."""

    def __init__(self):
      """Initialize test helper instance."""
      self.leading_trivia = ["some_trivia"]

  node = Tree("dummy", [DummyNodeWithTrivia()])
  assert _get_trivia(node) == ["some_trivia"]
