"""Docstring."""

import pytest
from ml_switcheroo.core.cst.base import CSTNode, Trivia


class MockCST(CSTNode):
  """Docstring."""

  def _get_name(self):
    """Docstring."""
    return "Mock"

  def _get_fields(self):
    """Docstring."""
    return {"a": 1}


def test_cst_base_render():
  """Docstring."""
  t = Trivia(text=" ")
  m = MockCST(leading_trivia=[t], trailing_trivia=[t])
  # The default to_text raises NotImplementedError
  with pytest.raises(NotImplementedError):
    m.to_text()

  rep = repr(m)
  assert "Mock" in rep


def test_cst_base_trivia_convert():
  """Docstring."""
  m = MockCST(leading_trivia=" ")
  assert len(m.leading_trivia) == 1
  assert m.leading_trivia[0].text == " "

  m2 = MockCST(trailing_trivia=" ")
  assert len(m2.trailing_trivia) == 1
  assert m2.trailing_trivia[0].text == " "


def test_trivia_eq():
  """Docstring."""
  t1 = Trivia(text="a")
  t2 = Trivia(text="a")
  assert t1 == t2
  assert t1 != "a"


def test_cst_base_trivia_convert_none():
  """Docstring."""
  m = MockCST(leading_trivia=None, trailing_trivia=None)
  assert m.leading_trivia == []
  assert m.trailing_trivia == []


def test_cst_base_str():
  """Docstring."""

  class MockStr(MockCST):
    """Docstring."""

    def to_text(self):
      """Docstring."""
      return "MockStr"

  m = MockStr()
  assert str(m) == "MockStr"


def test_cst_visitor():
  """Docstring."""
  from ml_switcheroo.core.cst.base import CSTVisitor

  class MockVisitor(CSTVisitor):
    """Docstring."""

    def visit_MockCST(self, node):
      """Docstring."""
      self.visited = True

  v = MockVisitor()
  m = MockCST()
  v.visit(m)
  assert getattr(v, "visited", False)


def test_cst_visitor_generic():
  """Docstring."""
  from ml_switcheroo.core.cst.base import CSTVisitor

  class MockVisitor(CSTVisitor):
    """Docstring."""

    def __init__(self):
      """Docstring."""
      self.generic_visited = False

    def generic_visit(self, node):
      """Docstring."""
      self.generic_visited = True

  v = MockVisitor()
  m = MockCST()
  v.visit(m)
  assert v.generic_visited


def test_cst_visitor_generic_fields():
  """Docstring."""
  from ml_switcheroo.core.cst.base import CSTVisitor
  from dataclasses import dataclass, field
  from typing import List

  @dataclass
  class NodeA(CSTNode):
    """Docstring."""

    child: CSTNode = None
    children: List[CSTNode] = field(default_factory=list)
    other: int = 1

  class MockVisitor(CSTVisitor):
    """Docstring."""

    def __init__(self):
      """Docstring."""
      self.visited_count = 0

    def generic_visit(self, node):
      """Docstring."""
      self.visited_count += 1
      super().generic_visit(node)

  v = MockVisitor()
  root = NodeA()
  root.child = NodeA()
  root.children = [NodeA(), NodeA()]
  v.visit(root)
  assert v.visited_count == 4


def test_cst_transformer_generic_fields():
  """Docstring."""
  from ml_switcheroo.core.cst.base import CSTTransformer
  from dataclasses import dataclass, field
  from typing import List

  @dataclass
  class NodeA(CSTNode):
    """Docstring."""

    child: CSTNode = None
    children: List[CSTNode] = field(default_factory=list)
    other: int = 1

  class MockTransformer(CSTTransformer):
    """Docstring."""

    def transform_NodeA(self, node):
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
  res = t.transform(root)
  assert res.child is child1
  assert res.children[0] is child2
  assert res.children[1] == "not a node"
