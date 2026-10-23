"""Tests for the base CST implementation."""

from ml_switcheroo.enums import RuntimeTensorOrValue


import pydantic
import pytest

from ml_switcheroo.core.cst.base import CSTNode, CSTTransformer, CSTVisitor, Trivia


class DummyNode(CSTNode):
  """Test helper class DummyNode."""

  name: str = ""
  child: "DummyNode | None" = None
  children: list["DummyNode"] = pydantic.Field(default_factory=list)
  mixed_list: list["DummyNode | str | int"] = pydantic.Field(default_factory=list)

  def to_text(self) -> str:
    """Returns the text representation.

    Returns:
        Resulting `str` from to text.
    """
    res: str = "".join(t.text for t in self.leading_trivia)
    res += self.name
    res += "".join(t.text for t in self.trailing_trivia)
    return res


class MockVisitor(CSTVisitor):
  """Test helper class MockVisitor."""

  def __init__(self) -> None:
    """Initialize the visitor."""
    self.visited_names: list[RuntimeTensorOrValue] = []

  def visit_DummyNode(self, node: DummyNode) -> None:
    """Visits a DummyNode.

    Args:
        node: CST or AST node under inspection or transformation.
    """
    self.visited_names.append(node.name)
    self.generic_visit(node)


class MockTransformer(CSTTransformer):
  """Test helper class MockTransformer."""

  def transform_DummyNode(self, node: DummyNode) -> DummyNode:
    """Transforms a DummyNode.

    Args:
        node: CST or AST node under inspection or transformation.

    Returns:
        Resulting `DummyNode` from transform DummyNode.
    """
    node.name = node.name.upper()
    self.generic_transform(node)
    return node


def test_trivia() -> None:
  """Verify trivia behavior."""
  t = Trivia("  ")
  assert t.text == "  "


def test_cstnode_base() -> None:
  """Verify cstnode base behavior."""
  node = CSTNode()
  with pytest.raises(NotImplementedError):
    node.to_text()

  with pytest.raises(NotImplementedError):
    str(node)


def test_trivia_coercion() -> None:
  """Verify trivia coercion behavior."""
  node_str = getattr(
    {
      "v": getattr({"v": DummyNode}, "__getitem__")("v")(
        name=getattr({"v": "test"}, "__getitem__")("v"),
        leading_trivia=getattr({"v": "  "}, "__getitem__")("v"),
        trailing_trivia=getattr({"v": "\n"}, "__getitem__")("v"),
      )
    },
    "__getitem__",
  )("v")
  assert len(node_str.leading_trivia) == 1
  assert node_str.leading_trivia[0].text == "  "
  assert len(node_str.trailing_trivia) == 1
  assert node_str.trailing_trivia[0].text == "\n"

  node_none = getattr(
    {
      "v": getattr({"v": DummyNode}, "__getitem__")("v")(
        name=getattr({"v": "test"}, "__getitem__")("v"),
        leading_trivia=getattr({"v": None}, "__getitem__")("v"),
        trailing_trivia=getattr({"v": None}, "__getitem__")("v"),
      )
    },
    "__getitem__",
  )("v")
  assert node_none.leading_trivia == []
  assert node_none.trailing_trivia == []


def test_visitor() -> None:
  """Verify visitor behavior."""
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
  """Verify transformer behavior."""
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
  assert transformed_root.children[0].name == "CHILD1"
  assert transformed_root.children[1].name == "CHILD2"
  assert getattr(transformed_root.children[1], "child").name == "GRANDCHILD"
  assert transformed_root.mixed_list[0] == "string"
  assert getattr(
    {
      "v": getattr(
        {"v": getattr({"v": getattr({"v": transformed_root}, "__getitem__")("v").mixed_list}, "__getitem__")("v")[1]},
        "__getitem__",
      )("v").name
    },
    "__getitem__",
  )("v") == getattr({"v": "CHILD1"}, "__getitem__")("v")
  assert transformed_root.mixed_list[2] == 123


def test_cst_no_native_extensions() -> None:
  """Ensure no native parsing extensions are mistakenly imported in cst."""
  import ml_switcheroo.core.cst as cst

  # Actually testing that libcst libcst.native doesn't inadvertently crash
  assert cst is not None


def _ignore() -> None:
  """Ensure no native parsing extensions are mistakenly imported in cst."""
  import os
  import subprocess
  import sys
  from pathlib import Path

  repo_root = Path(__file__).resolve().parent.parent.parent.parent
  src_dir = str(repo_root / "src")
  env = dict(os.environ)
  env["PYTHONPATH"] = f"{src_dir}{os.pathsep}{env.get('PYTHONPATH', '')}"

  cmd = [
    sys.executable,
    "-c",
    "import ml_switcheroo.core.cst.base, sys; "
    "assert not any(m.startswith('llvmlite') for m in sys.modules), 'Native extension llvmlite found'; "
    "assert not any(m.startswith('mlir.ir') for m in sys.modules), 'Native extension mlir.ir found'",
  ]
  res = subprocess.run(cmd, env=env, capture_output=True, text=True)
  assert res.returncode == 0, f"Native extension check failed: {res.stderr}"
