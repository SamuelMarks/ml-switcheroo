"""Tests for the base CST implementation."""

from dataclasses import dataclass, field
from typing import List

import pytest

from ml_switcheroo.core.cst.base import CSTNode, CSTTransformer, CSTVisitor, Trivia


@dataclass
class DummyNode(CSTNode):
  """Docstring."""

  name: str = ""
  child: "CSTNode" = None
  children: List["CSTNode"] = field(default_factory=list)
  mixed_list: List["CSTNode"] = field(default_factory=list)

  def to_text(self) -> str:
    """Returns the text representation."""
    res = "".join(t.text for t in self.leading_trivia)
    res += self.name
    res += "".join(t.text for t in self.trailing_trivia)
    return res


class MockVisitor(CSTVisitor):
  """Docstring."""

  def __init__(self) -> None:
    """Initialize the visitor."""
    self.visited_names: List[str] = []

  def visit_DummyNode(self, node: DummyNode) -> None:
    """Visits a DummyNode."""
    self.visited_names.append(node.name)
    self.generic_visit(node)


class MockTransformer(CSTTransformer):
  """Docstring."""

  def transform_DummyNode(self, node: DummyNode) -> DummyNode:
    """Transforms a DummyNode."""
    node.name = node.name.upper()
    self.generic_transform(node)
    return node


def test_trivia() -> None:
  """Docstring."""
  t = Trivia("  ")
  assert t.text == "  "


def test_cstnode_base() -> None:
  """Docstring."""
  node = CSTNode()
  with pytest.raises(NotImplementedError):
    node.to_text()

  with pytest.raises(NotImplementedError):
    str(node)


def test_trivia_coercion() -> None:
  """Docstring."""
  node_str = DummyNode(name="test", leading_trivia="  ", trailing_trivia="\n")
  assert len(node_str.leading_trivia) == 1
  assert node_str.leading_trivia[0].text == "  "
  assert len(node_str.trailing_trivia) == 1
  assert node_str.trailing_trivia[0].text == "\n"

  node_none = DummyNode(name="test", leading_trivia=None, trailing_trivia=None)
  assert node_none.leading_trivia == []
  assert node_none.trailing_trivia == []


def test_visitor() -> None:
  """Docstring."""
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
  """Docstring."""
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
  assert transformed_root.mixed_list[1].name == "CHILD1"
  assert transformed_root.mixed_list[2] == 123


def test_cst_no_native_extensions() -> None:
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
