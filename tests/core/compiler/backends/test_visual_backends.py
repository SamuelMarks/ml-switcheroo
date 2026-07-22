"""Test suite for the Visual Backends module."""

from ml_switcheroo.core.graph import LogicalGraph, LogicalNode, LogicalEdge
from ml_switcheroo.core.compiler.backends.visual_backends import TikzBackend, LatexBackend


def create_sample_graph():
  """Creates sample graph."""
  graph = LogicalGraph("TestGraph")
  graph.nodes = [
    LogicalNode("in", "Input", {"shape": "[10]"}),
    LogicalNode("l1", "Linear", {"features": 20, "bias": True}),
    LogicalNode("func_relu", "func_relu", {"arg1": "1.0"}),
    LogicalNode("out", "Output", {}),
  ]
  graph.edges = [LogicalEdge("in", "l1"), LogicalEdge("l1", "func_relu"), LogicalEdge("func_relu", "out")]
  return graph


def create_disconnected_graph():
  """Creates disconnected graph."""
  graph = LogicalGraph("DisGraph")
  graph.nodes = [LogicalNode("in", "Input", {}), LogicalNode("out", "Output", {})]
  graph.edges = []
  return graph


def test_tikz_backend_empty():
  """Verifies the behavior of TikZ backend empty."""
  backend = TikzBackend()
  graph = LogicalGraph("Empty")
  res = backend.compile(graph)
  assert "begin{tikzpicture}" in res


def test_tikz_backend_sample():
  """Verifies the behavior of TikZ backend sample."""
  backend = TikzBackend()
  graph = create_sample_graph()
  res = backend.compile(graph)
  assert "begin{tikzpicture}" in res
  assert "Linear" in res


def test_tikz_backend_disconnected():
  """Verifies the behavior of TikZ backend disconnected."""
  backend = TikzBackend()
  graph = create_disconnected_graph()
  res = backend.compile(graph)
  assert "in" in res


def test_latex_backend_empty():
  """Verifies the behavior of LaTeX backend empty."""
  backend = LatexBackend()
  graph = LogicalGraph()
  res = backend.compile(graph)
  assert "documentclass" in res
  assert "Model" in res


def test_latex_backend_sample():
  """Verifies the behavior of LaTeX backend sample."""
  backend = LatexBackend()
  graph = create_sample_graph()
  res = backend.compile(graph)
  assert "documentclass" in res
  assert "TestGraph" in res
  assert "Linear" in res
  assert "Relu" in res


def test_latex_backend_custom():
  """Verifies the behavior of LaTeX backend custom."""
  backend = LatexBackend()
  graph = LogicalGraph("Custom")
  graph.nodes = [LogicalNode("in", "Input", {}), LogicalNode("output", "Output", {})]
  graph.edges = [LogicalEdge("in", "some.op.Missing"), LogicalEdge("some.op.Missing", "output")]
  res = backend.compile(graph)
  assert "Missing" in res


def test_latex_backend_no_out_edges():
  """Verifies the behavior of LaTeX backend no output edges."""
  backend = LatexBackend()
  graph = LogicalGraph("Custom")
  graph.nodes = [LogicalNode("in", "Input", {}), LogicalNode("some_mod.foo", "some_mod.foo", {})]
  graph.edges = [LogicalEdge("in", "some_mod.foo")]
  res = backend.compile(graph)
  assert "ReturnNode" not in res or "last_step" in res
