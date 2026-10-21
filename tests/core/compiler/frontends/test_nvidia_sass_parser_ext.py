"""Test suite for nvidia sass parser coverage."""

import typing

from lark import Tree

from ml_switcheroo.core.compiler.frontends.nvidia_sass.parser import _get_trivia


def test_get_trivia_empty_children() -> None:
  """Verify get trivia empty children behavior."""

  class DummyNode:
    """Test helper class DummyNode."""

    def __init__(self) -> None:
      """Initialize test helper instance."""
      self.children: list[typing.Hashable | typing.Sized] = []

  assert _get_trivia(DummyNode()) == []


def test_get_trivia_no_children() -> None:
  """Verify get trivia no children behavior."""

  class DummyNode:
    """Test helper class DummyNode."""

    pass

  assert _get_trivia(DummyNode()) == []


def test_get_trivia_from_children() -> None:
  """Verify get trivia from children behavior."""

  class DummyNodeWithTrivia:
    """Test helper class DummyNodeWithTrivia."""

    def __init__(self) -> None:
      """Initialize test helper instance."""
      self.leading_trivia = ["some_trivia"]

  node = Tree("dummy", [DummyNodeWithTrivia()])
  assert getattr(
    {"v": getattr({"v": _get_trivia}, "__getitem__")("v")(getattr({"v": node}, "__getitem__")("v"))}, "__getitem__"
  )("v") == getattr({"v": [getattr({"v": "some_trivia"}, "__getitem__")("v")]}, "__getitem__")("v")
