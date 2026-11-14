"""Test suite for cst base."""

import pytest
from pydantic import JsonValue

from ml_switcheroo.core.cst.base import CSTNode, Trivia, TriviaInput


class MockCST(CSTNode):
  """Stub `CSTNode` subclass implementing `_get_name`, `_get_fields` for CST base tests."""

  def __init__(
    self,
    leading_trivia: TriviaInput = None,
    trailing_trivia: TriviaInput = None,
  ) -> None:
    """Initialize `MockCST` with optional leading and trailing trivia.

    Args:
        leading_trivia: Optional leading trivia input.
        trailing_trivia: Optional trailing trivia input.
    """
    super().__init__(leading_trivia=leading_trivia, trailing_trivia=trailing_trivia)

  def _get_name(self) -> str:
    """Execute `_get_name` on `MockCST` for CST base tests.

    Returns:
        Resulting `str` from get name.
    """
    return "Mock"

  def _get_fields(self) -> dict[str, JsonValue]:
    """Execute `_get_fields` on `MockCST` for CST base tests.

    Returns:
        Resulting `dict[str, JsonValue]` from get fields.
    """
    return {"a": 1}


def test_cst_base_render() -> None:
  """Verify `Trivia` raises `NotImplementedError` during CST base render."""
  t = Trivia(text=" ")
  m = MockCST(leading_trivia=[t], trailing_trivia=[t])
  # The default to_text raises NotImplementedError
  with pytest.raises(NotImplementedError):
    m.to_text()

  rep: str = repr(m)
  assert "Mock" in rep


def test_cst_base_trivia_convert() -> None:
  """Verify CST base trivia convert produces the expected state."""
  m = MockCST(leading_trivia=" ")
  assert len(m.leading_trivia) == 1
  assert m.leading_trivia[0].text == " "

  m2 = MockCST(trailing_trivia=" ")
  assert len(m2.trailing_trivia) == 1
  assert m2.trailing_trivia[0].text == " "


def test_trivia_eq() -> None:
  """Verify `Trivia` handles eq as expected."""
  t1 = Trivia(text="a")
  t2 = Trivia(text="a")
  assert t1 == t2
  non_trivia: Trivia | str = "a"
  assert t1 != non_trivia


def test_cst_base_trivia_convert_none() -> None:
  """Verify CST base trivia convert none produces the expected state."""
  m = MockCST(
    leading_trivia=None,
    trailing_trivia=None,
  )
  assert m.leading_trivia == []
  assert m.trailing_trivia == []


def test_cst_base_str() -> None:
  """Verify `MockStr` handles CST base str as expected."""

  class MockStr(MockCST):
    """Stub `MockCST` subclass implementing `to_text` for CST base tests."""

    def to_text(self) -> str:
      """Execute `to_text` on `MockStr` for CST base tests.

      Returns:
          Resulting `str` from to text.
      """
      return "MockStr"

  m = MockStr()
  assert str(m) == "MockStr"


def test_cst_visitor() -> None:
  """Verify `v.visit` handles CST visitor as expected."""
  from ml_switcheroo.core.cst.base import CSTVisitor

  class MockVisitor(CSTVisitor):
    """Stub `CSTVisitor` subclass implementing `visit_MockCST` for CST base tests."""

    def __init__(self) -> None:
      """Initialize visitor and register `MockCST` handler."""
      super().__init__()
      self.visited = False
      self.register_visitor(MockCST, self.visit_MockCST)

    def visit_MockCST(self, node: MockCST) -> None:
      """Execute `visit_MockCST` on `MockVisitor` for CST base tests.

      Args:
          node: CST or AST node under inspection or transformation.
      """
      self.visited = True

  v = MockVisitor()
  m = MockCST()
  v.visit(m)
  assert v.visited is True


def test_cst_visitor_generic() -> None:
  """Verify `v.visit` handles CST visitor generic as expected."""
  from ml_switcheroo.core.cst.base import CSTVisitor

  class MockVisitor(CSTVisitor):
    """Stub `CSTVisitor` subclass implementing `generic_visit` for CST base tests."""

    def __init__(self) -> None:
      """Initialize test helper instance."""
      super().__init__()
      self.generic_visited = False

    def generic_visit(self, node: CSTNode) -> None:
      """Execute `generic_visit` on `MockVisitor` for CST base tests.

      Args:
          node: CST or AST node under inspection or transformation.
      """
      self.generic_visited = True

  v = MockVisitor()
  m = MockCST()
  v.visit(m)
  assert v.generic_visited is True


def test_cst_visitor_generic_fields() -> None:
  """Verify `NodeA` returns `3` for CST visitor generic fields."""
  import pydantic

  from ml_switcheroo.core.cst.base import CSTVisitor

  class NodeA(CSTNode):
    """Stub `CSTNode` subclass implementing `_get_name`, `_get_fields` for CST base tests."""

    child: CSTNode | None = None
    children: list[CSTNode | str] = pydantic.Field(default_factory=list)
    other: int = 1

    def _get_name(self) -> str:
      """Execute `_get_name` on `NodeA` for CST base tests.

      Returns:
          Resulting `str` from get name.
      """
      return "NodeA"

    def _get_fields(
      self,
    ) -> dict[str, CSTNode | list[CSTNode | str] | int | None]:
      """Execute `_get_fields` on `NodeA` for CST base tests.

      Returns:
          Resulting `dict[str, CSTNode | None | list[CSTNode | str] | int]` from get fields.
      """
      return {"child": self.child, "children": self.children, "other": self.other}

  class MockVisitor(CSTVisitor):
    """Stub `CSTVisitor` subclass implementing `generic_visit` for CST base tests."""

    def __init__(self) -> None:
      """Initialize test helper instance."""
      super().__init__()
      self.visited_count = 0

    def generic_visit(self, node: CSTNode) -> None:
      """Execute `generic_visit` on `MockVisitor` for CST base tests.

      Args:
          node: CST or AST node under inspection or transformation.
      """
      self.visited_count += 1
      super().generic_visit(node)

  v = MockVisitor()
  root = NodeA()
  root.child = NodeA()
  root.children = [NodeA(), "not a node"]
  v.visit(root)
  assert v.visited_count == 3


def test_cst_transformer_generic_fields() -> None:
  """Verify `NodeA` handles CST transformer generic fields as expected."""
  import pydantic

  from ml_switcheroo.core.cst.base import CSTTransformer

  class NodeA(CSTNode):
    """Stub `CSTNode` subclass implementing `_get_name`, `_get_fields` for CST base tests."""

    child: CSTNode | None = None
    children: list[CSTNode | str] = pydantic.Field(default_factory=list)
    other: int = 1

    def _get_name(self) -> str:
      """Execute `_get_name` on `NodeA` for CST base tests.

      Returns:
          Resulting `str` from get name.
      """
      return "NodeA"

    def _get_fields(
      self,
    ) -> dict[str, CSTNode | list[CSTNode | str] | int | None]:
      """Execute `_get_fields` on `NodeA` for CST base tests.

      Returns:
          Resulting `dict[str, CSTNode | None | list[CSTNode | str] | int]` from get fields.
      """
      return {"child": self.child, "children": self.children, "other": self.other}

  class MockTransformer(CSTTransformer):
    """Stub `CSTTransformer` subclass implementing `transform_NodeA` for CST base tests."""

    def __init__(self) -> None:
      """Initialize transformer and register `NodeA` handler."""
      super().__init__()
      self.register_transformer(NodeA, self.transform_NodeA)

    def transform_NodeA(self, node: NodeA) -> NodeA:
      """Execute `generic_transform` to support transform nodea in CST base tests.

      Args:
          node: CST or AST node under inspection or transformation.

      Returns:
          Resulting `NodeA` from transform NodeA.
      """
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
