"""Tests for visual_latex.py."""

import ml_switcheroo.core.compiler.backends.visual_latex as visual_latex
from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode, LogicalEdge


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


def test_latex_backend_basic():
  """Verifies the behavior of latex backend basic."""
  backend = visual_latex.LatexBackend()
  graph = LogicalGraph("TestGraph")
  graph.nodes = [
    LogicalNode("in", "Input", {}),
    LogicalNode("out", "Output", {}),
    LogicalNode("comp", "Dense", {}),
  ]
  graph.edges = [LogicalEdge("in", "comp"), LogicalEdge("comp", "out")]
  res = backend.compile(graph)
  assert "documentclass" in res
  assert "TestGraph" in res
  assert "Dense" in res


def test_latex_backend_empty():
  """Verifies the behavior of LaTeX backend empty."""
  backend = visual_latex.LatexBackend()
  graph = LogicalGraph()
  res = backend.compile(graph)
  assert "documentclass" in res
  assert "Model" in res


def test_latex_backend_sample():
  """Verifies the behavior of LaTeX backend sample."""
  backend = visual_latex.LatexBackend()
  graph = create_sample_graph()
  res = backend.compile(graph)
  assert "documentclass" in res
  assert "TestGraph" in res
  assert "Linear" in res
  assert "Relu" in res


def test_latex_backend_custom():
  """Verifies the behavior of LaTeX backend custom."""
  backend = visual_latex.LatexBackend()
  graph = LogicalGraph("Custom")
  graph.nodes = [
    LogicalNode("in", "Input", {}),
    LogicalNode("some.op.Missing", "some.op.Missing", {}),
    LogicalNode("output", "Output", {}),
  ]
  graph.edges = [LogicalEdge("in", "some.op.Missing"), LogicalEdge("some.op.Missing", "output")]
  res = backend.compile(graph)
  assert "Missing" in res


def test_latex_backend_no_out_edges():
  """Verifies the behavior of LaTeX backend no output edges."""
  backend = visual_latex.LatexBackend()
  graph = LogicalGraph("Custom")
  graph.nodes = [LogicalNode("in", "Input", {}), LogicalNode("some_mod.foo", "some_mod.foo", {})]
  graph.edges = [LogicalEdge("in", "some_mod.foo")]
  res = backend.compile(graph)
  assert "ReturnNode" not in res or "last_step" in res


def test_latex_backend_output_node_bypass():
  """Test output node logic and func_ ignoring in state_registry."""
  backend = visual_latex.LatexBackend()
  graph = LogicalGraph("Custom")
  graph.nodes = [
    LogicalNode("in", "Input", {}),
    LogicalNode("func_foo", "my.module.Foo", {"non_arg": 123}),
    LogicalNode("Output", "Output", {}),
  ]
  graph.edges = [
    LogicalEdge("in", "func_foo"),
    LogicalEdge("in", "func_foo"),  # Duplicate edge to hit visited_ops continue
    LogicalEdge("func_foo", "Output"),
  ]
  res = backend.compile(graph)
  assert "Foo" in res
  assert "non_arg=123" in res


def test_visual_latex_no_node_data():
  # Hit 125->131
  """Test visual latex no node data."""
  from ml_switcheroo.core.compiler.backends.visual_latex import LatexBackend
  from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalEdge

  g = LogicalGraph("Test")
  # A graph with an edge but no nodes in graph.nodes.
  # target_id will not be found in node_dict, so node_data = None
  g.edges.append(LogicalEdge("in", "target"))
  backend = LatexBackend()
  code = backend.compile(g)
  assert "op_target" in code


def test_visual_latex_clean_type_no_dot_no_func():
  # Hit 135->137
  """Test visual latex clean type no dot no func."""
  from ml_switcheroo.core.compiler.backends.visual_latex import LatexBackend
  from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode, LogicalEdge

  g = LogicalGraph("Test")
  g.nodes.append(LogicalNode("in", "Input"))
  g.nodes.append(LogicalNode("target", "simple"))
  g.edges.append(LogicalEdge("in", "target"))
  backend = LatexBackend()
  code = backend.compile(g)
  assert "op_target = Simple" in code or "op_target" in code
