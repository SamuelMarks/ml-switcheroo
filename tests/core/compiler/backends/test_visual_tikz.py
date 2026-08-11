"""Tests for visual_tikz.py."""

import ml_switcheroo.core.compiler.backends.visual_tikz as visual_tikz
from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode, LogicalEdge


def test_tikz_backend_empty():
  """Verifies the behavior of TikZ backend empty."""
  backend = visual_tikz.TikzBackend()
  graph = LogicalGraph("Empty")
  res = backend.compile(graph)
  assert "begin{tikzpicture}" in res


def test_tikz_backend_pure_cycle():
  """Test cycle with no input nodes to hit not queue and graph.nodes branch."""
  backend = visual_tikz.TikzBackend()
  graph = LogicalGraph("CycleOnly")
  graph.nodes = [
    LogicalNode("n1", "Middle", {}),
    LogicalNode("n2", "Output", {}),
  ]
  graph.edges = [
    LogicalEdge("n1", "n2"),
    LogicalEdge("n2", "n1"),  # Pure cycle
  ]
  res = backend.compile(graph)
  assert "begin{tikzpicture}" in res


def test_tikz_backend_unconnected_cycle():
  """Test cycle in disconnected component to hit missing node from ranks."""
  backend = visual_tikz.TikzBackend()
  graph = LogicalGraph("IsolatedCycle")
  graph.nodes = [
    LogicalNode("in", "Input", {"shape": "[10]"}),
    LogicalNode("out", "Output", {}),
    LogicalNode("n1", "Isolated1", {}),
    LogicalNode("n2", "Isolated2", {}),
  ]
  graph.edges = [
    LogicalEdge("in", "out"),  # Normal path
    LogicalEdge("n1", "n2"),
    LogicalEdge("n2", "n1"),  # Isolated cycle
  ]
  res = backend.compile(graph)
  assert "Isolated1" in res


def test_visual_tikz_rank_existing_higher():
  # Hit 116->115
  """Test visual tikz rank existing higher."""
  from ml_switcheroo.core.compiler.backends.visual_tikz import TikzBackend
  from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode, LogicalEdge

  g = LogicalGraph("Test")
  g.nodes.append(LogicalNode("A", "Input"))
  g.nodes.append(LogicalNode("B", "Input"))
  g.nodes.append(LogicalNode("C", "Linear"))
  # A -> C, B -> C
  g.edges.append(LogicalEdge("A", "C"))
  g.edges.append(LogicalEdge("B", "C"))

  backend = TikzBackend()
  backend._calculate_layout(g)
