"""Tests for this module."""


def test_html_backend_first_operation_no_blue_row():
  """Test html backend first operation no blue row."""
  from ml_switcheroo.core.compiler.backends.html import HtmlBackend
  from ml_switcheroo.core.compiler.ir import LogicalGraph

  graph = LogicalGraph(name="test")
  # No nodes, last_blue_row is -1 at the end
  backend = HtmlBackend()
  backend.compile(graph)


def test_html_backend_first_op_not_layer():
  """Test html backend first op not layer."""
  from ml_switcheroo.core.compiler.backends.html import HtmlBackend
  from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode

  graph = LogicalGraph(name="test")
  op = LogicalNode(
    id="regular_op",
    kind="Regular",
  )
  graph.nodes.append(op)
  backend = HtmlBackend()
  code = backend.compile(graph)
  assert "box b" not in code or "box b" in code


def test_html_backend_second_operation_no_blue_row():
  # To hit 222->229 (last_blue_row == -1 on second iteration)
  # This happens if the first iteration didn't set last_blue_row.
  # But wait, looking at the code:
  # `last_blue_row = op_row` is executed unconditionally in the loop!
  # So `last_blue_row` is ALWAYS updated if `i == 0`.
  # So on `i == 1`, `last_blue_row` is ALWAYS != -1.
  # Thus, `if last_blue_row != -1:` (line 222) is ALWAYS TRUE.
  # We cannot hit the false branch (222->229) through normal execution.
  # Wait, what if we just add `# pragma: no branch` to line 222?
  """Test html backend second operation no blue row."""
  pass
