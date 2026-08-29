"""Module docstring."""

from lark import Tree

from ml_switcheroo.core.compiler.frontends.sass.parser import _get_trivia


def test_get_trivia_empty_children():
  """Docstring."""

  class DummyNode:
    def __init__(self):
      self.children = []

  assert _get_trivia(DummyNode()) == []


def test_get_trivia_no_children():
  """Docstring."""

  class DummyNode:
    pass

  assert _get_trivia(DummyNode()) == []


def test_get_trivia_from_children():
  """Docstring."""

  class DummyNodeWithTrivia:
    def __init__(self):
      self.leading_trivia = ["some_trivia"]

  node = Tree("dummy", [DummyNodeWithTrivia()])
  assert _get_trivia(node) == ["some_trivia"]
