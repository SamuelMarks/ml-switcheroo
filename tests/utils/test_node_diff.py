"""Test suite for the Node Diff module."""

import libcst as cst

from ml_switcheroo.utils.node_diff import capture_node_source, diff_nodes


def test_capture_simple_call() -> None:
  """Verifies the behavior of capture simple call."""
  node: cst.Call = cst.Call(func=cst.Name("my_func"), args=[cst.Arg(cst.Integer("1"))])
  source: str = capture_node_source(node)
  assert "my_func(1)" in source


def test_capture_complex_assignment() -> None:
  """Verifies the behavior of capture complex assignment."""
  target: cst.AssignTarget = cst.AssignTarget(target=cst.Name("x"))
  node: cst.Assign = cst.Assign(targets=[target], value=cst.Integer("10"))
  source: str = capture_node_source(node)
  assert "x = 10" in source


def test_diff_nodes_detection() -> None:
  """Verifies the behavior of diff nodes detection."""
  node_a: cst.Call = cst.Call(func=cst.Name("foo"))
  node_b: cst.Call = cst.Call(func=cst.Name("bar"))
  before: str
  after: str
  changed: bool
  before, after, changed = diff_nodes(node_a, node_b)
  assert changed is True
  assert before == "foo()"
  assert after == "bar()"


def test_diff_nodes_no_change() -> None:
  """Verifies the behavior of diff nodes no change."""
  node_a: cst.Call = cst.Call(func=cst.Name("foo"))
  node_b: cst.Call = cst.Call(func=cst.Name("foo"))
  changed: bool
  _, _, changed = diff_nodes(node_a, node_b)
  assert changed is False


def test_capture_fallback() -> None:
  """Verifies the behavior of capture fallback."""
  res: str = capture_node_source("NotANode")
  assert "<Unrepresentable Node: str>" in res
