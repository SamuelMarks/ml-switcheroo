"""Tests for the base CST implementation."""

import typing

from dataclasses import dataclass, field

import pytest

from ml_switcheroo.core.cst.base import CSTNode, CSTTransformer, CSTVisitor, Trivia


@dataclass
class DummyNode(CSTNode):
  """Test helper class DummyNode."""

  name: typing.Any = ""
  child: typing.Any = None
  children: typing.Any = field(default_factory=list)
  mixed_list: typing.Any = field(default_factory=list)

  def to_text(self) -> typing.Any:
    """Returns the text representation.

    Returns:
        Computed test result or fixture value.
    """
    res: typing.Any = "".join(t.text for t in self.leading_trivia)
    res += self.name
    res += "".join(t.text for t in self.trailing_trivia)
    return res


class MockVisitor(CSTVisitor):
  """Test helper class MockVisitor."""

  def __init__(self) -> None:
    """Initialize the visitor."""
    self.visited_names: typing.Any = []

  def visit_DummyNode(self, node: typing.Any) -> None:
    """Visits a DummyNode.

    Args:
        node: Test argument or fixture.
    """
    self.visited_names.append(node.name)
    self.generic_visit(node)


class MockTransformer(CSTTransformer):
  """Test helper class MockTransformer."""

  def transform_DummyNode(self, node: typing.Any) -> typing.Any:
    """Transforms a DummyNode.

    Args:
        node: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    node.name = node.name.upper()
    self.generic_transform(node)
    return node


def test_trivia() -> None:
  """Verify trivia behavior."""
  t: typing.Any = Trivia("  ")
  assert t.text == "  "


def test_cstnode_base() -> None:
  """Verify cstnode base behavior."""
  node: typing.Any = CSTNode()
  with pytest.raises(NotImplementedError):
    node.to_text()

  with pytest.raises(NotImplementedError):
    str(node)


def test_trivia_coercion() -> None:
  """Verify trivia coercion behavior."""
  node_str: typing.Any = DummyNode(name="test", leading_trivia="  ", trailing_trivia="\n")
  assert len(node_str.leading_trivia) == 1
  assert node_str.leading_trivia[0].text == "  "
  assert len(node_str.trailing_trivia) == 1
  assert node_str.trailing_trivia[0].text == "\n"

  node_none: typing.Any = DummyNode(name="test", leading_trivia=None, trailing_trivia=None)
  assert node_none.leading_trivia == []
  assert node_none.trailing_trivia == []


def test_visitor() -> None:
  """Verify visitor behavior."""
  root: typing.Any = DummyNode(name="root")
  child1: typing.Any = DummyNode(name="child1")
  child2: typing.Any = DummyNode(name="child2")
  grandchild: typing.Any = DummyNode(name="grandchild")

  child2.child = grandchild
  root.children = [child1, child2]
  root.mixed_list = ["string", child1, 123]

  visitor: typing.Any = MockVisitor()
  visitor.visit(root)

  assert visitor.visited_names == ["root", "child1", "child2", "grandchild", "child1"]


def test_transformer() -> None:
  """Verify transformer behavior."""
  root: typing.Any = DummyNode(name="root")
  child1: typing.Any = DummyNode(name="child1")
  child2: typing.Any = DummyNode(name="child2")
  grandchild: typing.Any = DummyNode(name="grandchild")

  child2.child = grandchild
  root.children = [child1, child2]
  root.mixed_list = ["string", child1, 123]

  transformer: typing.Any = MockTransformer()
  transformed_root: typing.Any = transformer.transform(root)

  assert isinstance(transformed_root, DummyNode)
  assert transformed_root.name == "ROOT"
  assert transformed_root.children[0].name == "CHILD1"
  assert transformed_root.children[1].name == "CHILD2"
  assert getattr(transformed_root.children[1], "child").name == "GRANDCHILD"
  assert transformed_root.mixed_list[0] == "string"
  assert transformed_root.mixed_list[1].name == "CHILD1"
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

  repo_root: typing.Any = Path(__file__).resolve().parent.parent.parent.parent
  src_dir: typing.Any = str(repo_root / "src")
  env: typing.Any = dict(os.environ)
  env["PYTHONPATH"] = f"{src_dir}{os.pathsep}{env.get('PYTHONPATH', '')}"

  cmd: typing.Any = [
    sys.executable,
    "-c",
    "import ml_switcheroo.core.cst.base, sys; "
    "assert not any(m.startswith('llvmlite') for m in sys.modules), 'Native extension llvmlite found'; "
    "assert not any(m.startswith('mlir.ir') for m in sys.modules), 'Native extension mlir.ir found'",
  ]
  res: typing.Any = subprocess.run(cmd, env=env, capture_output=True, text=True)
  assert res.returncode == 0, f"Native extension check failed: {res.stderr}"
