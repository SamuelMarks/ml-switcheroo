"""Test suite for the Node Diff module."""

import libcst as _cst
from libcst._nodes.internal import CodegenState

from ml_switcheroo.utils.node_diff import capture_node_source, diff_nodes

cst = _cst


def test_capture_simple_call() -> None:
  """Verify `Call` emits `my_func(1)` for capture simple call."""
  node = cst.Call(func=cst.Name("my_func"), args=[cst.Arg(cst.Integer("1"))])
  source = capture_node_source(node)
  assert "my_func(1)" in source


def test_capture_complex_assignment() -> None:
  """Verify `AssignTarget` emits `x = 10` for capture complex assignment."""
  target = cst.AssignTarget(target=cst.Name("x"))
  node = cst.Assign(targets=[target], value=cst.Integer("10"))
  source = capture_node_source(node)
  assert "x = 10" in source


def test_diff_nodes_detection() -> None:
  """Verify `Call` handles diff nodes detection as expected."""
  node_a = cst.Call(func=cst.Name("foo"))
  node_b = cst.Call(func=cst.Name("bar"))
  before, after, changed = diff_nodes(node_a, node_b)
  assert changed is True
  assert before == "foo()"
  assert after == "bar()"


def test_diff_nodes_no_change() -> None:
  """Verify `Call` handles diff nodes no change as expected."""
  node_a = cst.Call(func=cst.Name("foo"))
  node_b = cst.Call(func=cst.Name("foo"))
  _, _, changed = diff_nodes(node_a, node_b)
  assert changed is False


def test_capture_fallback() -> None:
  """Verify `capture_node_source` handles capture fallback as expected."""

  class UnrepresentableNode(_cst.CSTNode):
    """Unrepresentable CST node for testing fallback."""

    def _visit_and_replace_children(self, visitor: _cst.CSTVisitorT) -> "UnrepresentableNode":
      """Visit children stub.

      Args:
          visitor: CST visitor.

      Returns:
          Self.
      """
      return self

    def _codegen_impl(self, state: CodegenState) -> None:
      """Codegen stub that raises an error.

      Args:
          state: Codegen state.

      Raises:
          RuntimeError: Always raised to trigger fallback.
      """
      raise RuntimeError("Cannot render")

  res = capture_node_source(UnrepresentableNode())
  assert "<Unrepresentable Node: UnrepresentableNode>" in res
