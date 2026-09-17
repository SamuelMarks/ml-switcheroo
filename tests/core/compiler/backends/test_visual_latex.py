"""Tests for visual_latex.py."""

import ml_switcheroo.core.compiler.backends.visual_latex as visual_latex
from ml_switcheroo.core.compiler.ir import LogicalEdge, LogicalGraph, LogicalNode


def create_sample_graph() -> LogicalGraph:
  """Creates sample graph."""
  nodes = {
    "in": LogicalNode("in", op_type="Input", attributes={"shape": "[10]"}),
    "l1": LogicalNode("l1", op_type="Linear", attributes={"features": "20", "bias": "True"}),
    "func_relu": LogicalNode("func_relu", op_type="func_relu", attributes={"arg1": "1.0"}),
    "out": LogicalNode("out", op_type="Output", attributes={}),
  }
  edges = [LogicalEdge("in", "l1"), LogicalEdge("l1", "func_relu"), LogicalEdge("func_relu", "out")]
  return LogicalGraph("TestGraph", nodes=nodes, edges=edges)


def test_latex_backend_basic() -> None:
  """Verifies the behavior of latex backend basic."""
  backend = visual_latex.LatexBackend()
  nodes = {
    "in": LogicalNode("in", op_type="Input", attributes={}),
    "out": LogicalNode("out", op_type="Output", attributes={}),
    "comp": LogicalNode("comp", op_type="Dense", attributes={}),
  }
  edges = [LogicalEdge("in", "comp"), LogicalEdge("comp", "out")]
  graph = LogicalGraph("TestGraph", nodes=nodes, edges=edges)
  res: str = backend.compile(graph)
  assert "documentclass" in res
  assert "TestGraph" in res
  assert "Dense" in res


def test_latex_backend_empty() -> None:
  """Verifies the behavior of LaTeX backend empty."""
  backend = visual_latex.LatexBackend()
  graph = LogicalGraph()
  res: str = backend.compile(graph)
  assert "documentclass" in res
  assert "Model" in res


def test_latex_backend_sample() -> None:
  """Verifies the behavior of LaTeX backend sample."""
  backend = visual_latex.LatexBackend()
  graph: LogicalGraph = create_sample_graph()
  res: str = backend.compile(graph)
  assert "documentclass" in res
  assert "TestGraph" in res
  assert "Linear" in res
  assert "Relu" in res


def test_latex_backend_custom() -> None:
  """Verifies the behavior of LaTeX backend custom."""
  backend = visual_latex.LatexBackend()
  nodes = {
    "in": LogicalNode("in", op_type="Input", attributes={}),
    "some.op.Missing": LogicalNode("some.op.Missing", op_type="some.op.Missing", attributes={}),
    "output": LogicalNode("output", op_type="Output", attributes={}),
  }
  edges = [LogicalEdge("in", "some.op.Missing"), LogicalEdge("some.op.Missing", "output")]
  graph = LogicalGraph("Custom", nodes=nodes, edges=edges)
  res: str = backend.compile(graph)
  assert "Missing" in res


def test_latex_backend_no_out_edges() -> None:
  """Verifies the behavior of LaTeX backend no output edges."""
  backend = visual_latex.LatexBackend()
  nodes = {
    "in": LogicalNode("in", op_type="Input", attributes={}),
    "some_mod.foo": LogicalNode("some_mod.foo", op_type="some_mod.foo", attributes={}),
  }
  edges = [LogicalEdge("in", "some_mod.foo")]
  graph = LogicalGraph("Custom", nodes=nodes, edges=edges)
  res: str = backend.compile(graph)
  assert "ReturnNode" not in res or "last_step" in res


def test_latex_backend_output_node_bypass() -> None:
  """Verifies LaTeX backend handles nodes with multiple incoming edges and output bypass."""
  backend = visual_latex.LatexBackend()
  nodes = {
    "in1": LogicalNode("in1", op_type="Input", attributes={}),
    "in2": LogicalNode("in2", op_type="Input", attributes={}),
    "func_foo": LogicalNode("func_foo", op_type="my.module.Foo", attributes={"non_arg": "123"}),
    "Output": LogicalNode("Output", op_type="Output", attributes={}),
  }
  edges = [
    LogicalEdge("in1", "func_foo"),
    LogicalEdge("in2", "func_foo"),  # Duplicate target to hit visited_ops continue
    LogicalEdge("func_foo", "Output"),
  ]
  graph = LogicalGraph("Custom", nodes=nodes, edges=edges)
  res: str = backend.compile(graph)
  assert "Foo" in res
  assert "non_arg=123" in res


def test_visual_latex_no_node_data() -> None:
  """Docstring."""
  # Hit 125->131
  from ml_switcheroo.core.compiler.backends.visual_latex import LatexBackend
  from ml_switcheroo.core.compiler.ir import LogicalEdge, LogicalGraph

  g = LogicalGraph("Test", nodes={}, edges=[LogicalEdge("in", "target")])
  backend = LatexBackend()
  code: str = backend.compile(g)
  assert "op_target" in code


def test_visual_latex_clean_type_no_dot_no_func() -> None:
  """Docstring."""
  # Hit 135->137
  from ml_switcheroo.core.compiler.backends.visual_latex import LatexBackend
  from ml_switcheroo.core.compiler.ir import LogicalEdge, LogicalGraph, LogicalNode

  nodes = {
    "in": LogicalNode("in", op_type="Input"),
    "target": LogicalNode("target", op_type="simple"),
  }
  edges = [LogicalEdge("in", "target")]
  g = LogicalGraph("Test", nodes=nodes, edges=edges)
  backend = LatexBackend()
  code: str = backend.compile(g)
  assert "op_target = Simple" in code or "op_target" in code
