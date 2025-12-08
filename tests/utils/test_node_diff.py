"""
Tests for Node Diff Utility.
"""

import pytest
import libcst as cst
from ml_switcheroo.utils.node_diff import capture_node_source, diff_nodes


def test_capture_simple_call():
  """Verify source code capture for a detached Call node."""
  node = cst.Call(func=cst.Name("my_func"), args=[cst.Arg(cst.Integer("1"))])
  source = capture_node_source(node)

  # whitespace might vary depending on LibCST defaults, usually no spaces for detached
  assert "my_func(1)" in source


def test_capture_complex_assignment():
  """Verify capture of an assignment structure."""
  target = cst.AssignTarget(target=cst.Name("x"))
  node = cst.Assign(targets=[target], value=cst.Integer("10"))
  source = capture_node_source(node)
  assert "x = 10" in source


def test_diff_nodes_detection():
  """Verify diff logic returns correct boolean."""
  node_a = cst.Call(func=cst.Name("foo"))
  node_b = cst.Call(func=cst.Name("bar"))

  before, after, changed = diff_nodes(node_a, node_b)

  assert changed is True
  assert before == "foo()"
  assert after == "bar()"


def test_diff_nodes_no_change():
  """Verify no change is detected for identical structures."""
  node_a = cst.Call(func=cst.Name("foo"))
  node_b = cst.Call(func=cst.Name("foo"))

  _, _, changed = diff_nodes(node_a, node_b)
  assert changed is False


def test_capture_fallback():
  """Verify robust handling of unrepresentable nodes."""
  # It's hard to force LibCST to fail code generation on valid types,
  # but we can try passing an object that isn't a CSTNode.
  # capture_node_source calls code_for_node which expects CSTNode.

  # Pass a string instead of a Node
  res = capture_node_source("NotANode")  # type: ignore
  assert "<Unrepresentable Node: str>" in res
