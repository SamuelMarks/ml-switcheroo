"""Tests for visual_tikz.py."""

import ml_switcheroo.core.compiler.backends.visual_tikz as visual_tikz
from ml_switcheroo.core.compiler.ir import LogicalEdge, LogicalGraph, LogicalNode


def test_tikz_backend_empty() -> None:
  """Verifies the behavior of TikZ backend empty."""
  backend = visual_tikz.TikzBackend()
  graph = LogicalGraph("Empty")
  res: str = backend.compile(graph)
  assert "begin{tikzpicture}" in res


def test_tikz_backend_pure_cycle() -> None:
  """Docstring."""
  backend = visual_tikz.TikzBackend()
  nodes = {
    "n1": LogicalNode("n1", op_type="Middle", attributes={}),
    "n2": LogicalNode("n2", op_type="Output", attributes={}),
  }
  edges = [
    LogicalEdge("n1", "n2"),
    LogicalEdge("n2", "n1"),  # Pure cycle
  ]
  graph = LogicalGraph("CycleOnly", nodes=nodes, edges=edges)
  res: str = backend.compile(graph)
  assert "begin{tikzpicture}" in res


def test_tikz_backend_unconnected_cycle() -> None:
  """Docstring."""
  backend = visual_tikz.TikzBackend()
  nodes = {
    "in": LogicalNode("in", op_type="Input", attributes={"shape": "[10]"}),
    "out": LogicalNode("out", op_type="Output", attributes={}),
    "n1": LogicalNode("n1", op_type="Isolated1", attributes={}),
    "n2": LogicalNode("n2", op_type="Isolated2", attributes={}),
  }
  edges = [
    LogicalEdge("in", "out"),  # Normal path
    LogicalEdge("n1", "n2"),
    LogicalEdge("n2", "n1"),  # Isolated cycle
  ]
  graph = LogicalGraph("IsolatedCycle", nodes=nodes, edges=edges)
  res: str = backend.compile(graph)
  assert "Isolated1" in res


def test_visual_tikz_rank_existing_higher() -> None:
  """Docstring."""
  from ml_switcheroo.core.compiler.backends.visual_tikz import TikzBackend
  from ml_switcheroo.core.compiler.ir import LogicalEdge, LogicalGraph, LogicalNode

  nodes = {
    "A": LogicalNode("A", op_type="Input"),
    "B": LogicalNode("B", op_type="Input"),
    "C": LogicalNode("C", op_type="Linear"),
  }
  edges = [
    LogicalEdge("A", "C"),
    LogicalEdge("B", "C"),
  ]
  g = LogicalGraph("Test", nodes=nodes, edges=edges)

  backend = TikzBackend()
  backend._calculate_layout(g)
