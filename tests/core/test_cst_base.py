"""Docstring."""

import typing

import pytest

from ml_switcheroo.core.cst.base import CSTNode, Trivia


class MockCST(CSTNode):
  """Docstring."""

  def _get_name(self) -> str:
    """Docstring."""
    return "Mock"

  def _get_fields(self) -> dict[str, typing.Any]:
    """Docstring."""
    return {"a": 1}


def test_cst_base_render() -> None:
  """Docstring."""
  t = Trivia(text=" ")
  m = MockCST(leading_trivia=[t], trailing_trivia=[t])  # type: ignore
  # The default to_text raises NotImplementedError
  with pytest.raises(NotImplementedError):
    m.to_text()

  rep: str = repr(m)
  assert "Mock" in rep


def test_cst_base_trivia_convert() -> None:
  """Docstring."""
  m = MockCST(leading_trivia=" ")  # type: ignore
  assert len(m.leading_trivia) == 1
  assert m.leading_trivia[0].text == " "

  m2 = MockCST(trailing_trivia=" ")  # type: ignore
  assert len(m2.trailing_trivia) == 1
  assert m2.trailing_trivia[0].text == " "


def test_trivia_eq() -> None:
  """Docstring."""
  t1 = Trivia(text="a")
  t2 = Trivia(text="a")
  assert t1 == t2
  assert t1 != "a"


def test_cst_base_trivia_convert_none() -> None:
  """Docstring."""
  m = MockCST(leading_trivia=None, trailing_trivia=None)  # type: ignore
  assert m.leading_trivia == []
  assert m.trailing_trivia == []


def test_cst_base_str() -> None:
  """Docstring."""

  class MockStr(MockCST):
    """Docstring."""

    def to_text(self) -> str:
      """Docstring."""
      return "MockStr"

  m = MockStr()
  assert str(m) == "MockStr"


def test_cst_visitor() -> None:
  """Docstring."""
  from ml_switcheroo.core.cst.base import CSTVisitor

  class MockVisitor(CSTVisitor):
    """Docstring."""

    def visit_MockCST(self, node: CSTNode) -> None:
      """Docstring."""
      self.visited = True  # type: ignore

  v = MockVisitor()
  m = MockCST()
  v.visit(m)
  assert getattr(v, "visited", False)


def test_cst_visitor_generic() -> None:
  """Docstring."""
  from ml_switcheroo.core.cst.base import CSTVisitor

  class MockVisitor(CSTVisitor):
    """Docstring."""

    def __init__(self) -> None:
      """Docstring."""
      self.generic_visited = False

    def generic_visit(self, node: CSTNode) -> None:
      """Docstring."""
      self.generic_visited = True

  v = MockVisitor()
  m = MockCST()
  v.visit(m)
  assert v.generic_visited


def test_cst_visitor_generic_fields() -> None:
  """Docstring."""
  from dataclasses import dataclass, field
  from typing import List, Optional

  from ml_switcheroo.core.cst.base import CSTVisitor

  @dataclass
  class NodeA(CSTNode):
    """Docstring."""

    child: Optional[CSTNode] = None
    children: List[typing.Any] = field(default_factory=list)
    other: int = 1

    def _get_name(self) -> str:
      """Docstring."""
      return "NodeA"

    def _get_fields(self) -> dict[str, typing.Any]:
      """Docstring."""
      return {"child": self.child, "children": self.children, "other": self.other}

  class MockVisitor(CSTVisitor):
    """Docstring."""

    def __init__(self) -> None:
      """Docstring."""
      self.visited_count = 0

    def generic_visit(self, node: CSTNode) -> None:
      """Docstring."""
      self.visited_count += 1
      super().generic_visit(node)

  v = MockVisitor()
  root = NodeA()
  root.child = NodeA()
  root.children = [NodeA(), "not a node"]
  v.visit(root)
  assert v.visited_count == 3


def test_cst_transformer_generic_fields() -> None:
  """Docstring."""
  from dataclasses import dataclass, field
  from typing import List, Optional

  from ml_switcheroo.core.cst.base import CSTTransformer

  @dataclass
  class NodeA(CSTNode):
    """Docstring."""

    child: Optional[CSTNode] = None
    children: List[typing.Any] = field(default_factory=list)
    other: int = 1

    def _get_name(self) -> str:
      """Docstring."""
      return "NodeA"

    def _get_fields(self) -> dict[str, typing.Any]:
      """Docstring."""
      return {"child": self.child, "children": self.children, "other": self.other}

  class MockTransformer(CSTTransformer):
    """Docstring."""

    def transform_NodeA(self, node: NodeA) -> typing.Any:
      """Docstring."""
      if node.other == 1:
        return self.generic_transform(node)
      return node

  t = MockTransformer()
  root = NodeA(other=1)
  child1 = NodeA(other=2)
  child2 = NodeA(other=3)
  root.child = child1
  root.children = [child2, "not a node"]
  res: typing.Any = t.transform(root)
  assert res.child is child1
  assert res.children[0] is child2
  assert res.children[1] == "not a node"
