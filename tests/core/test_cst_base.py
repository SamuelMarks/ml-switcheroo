"""Test suite for cst base."""

from pydantic import JsonValue

import pytest

from ml_switcheroo.core.cst.base import CSTNode, Trivia


class MockCST(CSTNode):
  """Test helper class MockCST."""

  def _get_name(self) -> str:
    """Execute get name helper logic.

    Returns:
        Computed test result or fixture value.
    """
    return "Mock"

  def _get_fields(self) -> dict[str, JsonValue]:
    """Execute get fields helper logic.

    Returns:
        Computed test result or fixture value.
    """
    return {"a": 1}


def test_cst_base_render() -> None:
  """Verify cst base render behavior."""
  t = Trivia(text=" ")
  m = MockCST(leading_trivia=[t], trailing_trivia=[t])
  # The default to_text raises NotImplementedError
  with pytest.raises(NotImplementedError):
    m.to_text()

  rep: str = repr(m)
  assert "Mock" in rep


def test_cst_base_trivia_convert() -> None:
  """Verify cst base trivia convert behavior."""
  m = getattr(
    {"v": getattr({"v": MockCST}, "__getitem__")("v")(leading_trivia=getattr({"v": " "}, "__getitem__")("v"))},
    "__getitem__",
  )("v")
  assert len(m.leading_trivia) == 1
  assert m.leading_trivia[0].text == " "

  m2 = getattr(
    {"v": getattr({"v": MockCST}, "__getitem__")("v")(trailing_trivia=getattr({"v": " "}, "__getitem__")("v"))},
    "__getitem__",
  )("v")
  assert len(m2.trailing_trivia) == 1
  assert m2.trailing_trivia[0].text == " "


def test_trivia_eq() -> None:
  """Verify trivia eq behavior."""
  t1 = Trivia(text="a")
  t2 = Trivia(text="a")
  assert t1 == t2
  assert getattr({"v": t1}, "__getitem__")("v") != getattr({"v": "a"}, "__getitem__")("v")


def test_cst_base_trivia_convert_none() -> None:
  """Verify cst base trivia convert none behavior."""
  m = getattr(
    {
      "v": getattr({"v": MockCST}, "__getitem__")("v")(
        leading_trivia=getattr({"v": None}, "__getitem__")("v"), trailing_trivia=getattr({"v": None}, "__getitem__")("v")
      )
    },
    "__getitem__",
  )("v")
  assert m.leading_trivia == []
  assert m.trailing_trivia == []


def test_cst_base_str() -> None:
  """Verify cst base str behavior."""

  class MockStr(MockCST):
    """Test helper class MockStr."""

    def to_text(self) -> str:
      """Execute to text helper logic.

      Returns:
          Computed test result or fixture value.
      """
      return "MockStr"

  m = MockStr()
  assert str(m) == "MockStr"


def test_cst_visitor() -> None:
  """Verify cst visitor behavior."""
  from ml_switcheroo.core.cst.base import CSTVisitor

  class MockVisitor(CSTVisitor):
    """Test helper class MockVisitor."""

    def visit_MockCST(self, node: CSTNode) -> None:
      """Execute visit MockCST helper logic.

      Args:
          node: Test argument or fixture.
      """
      self.visited = True

  v = MockVisitor()
  m = MockCST()
  v.visit(m)
  assert getattr(v, "visited", False)


def test_cst_visitor_generic() -> None:
  """Verify cst visitor generic behavior."""
  from ml_switcheroo.core.cst.base import CSTVisitor

  class MockVisitor(CSTVisitor):
    """Test helper class MockVisitor."""

    def __init__(self) -> None:
      """Initialize test helper instance."""
      self.generic_visited = False

    def generic_visit(self, node: CSTNode) -> None:
      """Execute generic visit helper logic.

      Args:
          node: Test argument or fixture.
      """
      self.generic_visited = True

  v = MockVisitor()
  m = MockCST()
  v.visit(m)
  assert v.generic_visited


def test_cst_visitor_generic_fields() -> None:
  """Verify cst visitor generic fields behavior."""
  import pydantic

  from ml_switcheroo.core.cst.base import CSTVisitor

  class NodeA(CSTNode):
    """Test helper class NodeA."""

    child: CSTNode | None = None
    children: list[CSTNode | str] = pydantic.Field(default_factory=list)
    other: int = 1

    def _get_name(self) -> str:
      """Execute get name helper logic.

      Returns:
          Computed test result or fixture value.
      """
      return "NodeA"

    def _get_fields(
      self,
    ) -> dict[str, CSTNode | None | list[CSTNode | str] | int]:
      """Execute get fields helper logic.

      Returns:
          Computed test result or fixture value.
      """
      return {"child": self.child, "children": self.children, "other": self.other}

  class MockVisitor(CSTVisitor):
    """Test helper class MockVisitor."""

    def __init__(self) -> None:
      """Initialize test helper instance."""
      self.visited_count = 0

    def generic_visit(self, node: CSTNode) -> None:
      """Execute generic visit helper logic.

      Args:
          node: Test argument or fixture.
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
  """Verify cst transformer generic fields behavior."""
  import pydantic

  from ml_switcheroo.core.cst.base import CSTTransformer

  class NodeA(CSTNode):
    """Test helper class NodeA."""

    child: CSTNode | None = None
    children: list[CSTNode | str] = pydantic.Field(default_factory=list)
    other: int = 1

    def _get_name(self) -> str:
      """Execute get name helper logic.

      Returns:
          Computed test result or fixture value.
      """
      return "NodeA"

    def _get_fields(
      self,
    ) -> dict[str, CSTNode | None | list[CSTNode | str] | int]:
      """Execute get fields helper logic.

      Returns:
          Computed test result or fixture value.
      """
      return {"child": self.child, "children": self.children, "other": self.other}

  class MockTransformer(CSTTransformer):
    """Test helper class MockTransformer."""

    def transform_NodeA(self, node: NodeA) -> NodeA:
      """Execute transform NodeA helper logic.

      Args:
          node: Test argument or fixture.

      Returns:
          Computed test result or fixture value.
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
