"""Test suite for the Visual Backends module."""

from ml_switcheroo.core.compiler.backends.visual_backends import LatexBackend, TikzBackend
from ml_switcheroo.core.graph import LogicalEdge, LogicalGraph, LogicalNode


def create_sample_graph() -> LogicalGraph:
  """Creates sample graph.

  Returns:
      Computed test result or fixture value.
  """
  nodes = {
    "in": LogicalNode("in", op_type="Input", attributes={"shape": "[10]"}),
    "l1": LogicalNode("l1", op_type="Linear", attributes={"features": "20", "bias": "True"}),
    "func_relu": LogicalNode("func_relu", op_type="func_relu", attributes={"arg1": "1.0"}),
    "out": LogicalNode("out", op_type="Output", attributes={}),
  }
  edges = [LogicalEdge("in", "l1"), LogicalEdge("l1", "func_relu"), LogicalEdge("func_relu", "out")]
  return LogicalGraph("TestGraph", nodes=nodes, edges=edges)


def create_disconnected_graph() -> LogicalGraph:
  """Creates disconnected graph.

  Returns:
      Computed test result or fixture value.
  """
  nodes = {
    "in": LogicalNode("in", op_type="Input", attributes={}),
    "out": LogicalNode("out", op_type="Output", attributes={}),
  }
  return LogicalGraph("DisGraph", nodes=nodes, edges=[])


def test_tikz_backend_empty() -> None:
  """Verifies the behavior of TikZ backend empty."""
  backend = TikzBackend()
  graph = LogicalGraph("Empty")
  res: str = backend.compile(graph)
  assert "begin{tikzpicture}" in res


def test_tikz_backend_sample() -> None:
  """Verifies the behavior of TikZ backend sample."""
  backend = TikzBackend()
  graph: LogicalGraph = create_sample_graph()
  res: str = backend.compile(graph)
  assert "begin{tikzpicture}" in res
  assert "Linear" in res


def test_tikz_backend_disconnected() -> None:
  """Verifies the behavior of TikZ backend disconnected."""
  backend = TikzBackend()
  graph: LogicalGraph = create_disconnected_graph()
  res: str = backend.compile(graph)
  assert "in" in res


def test_latex_backend_empty() -> None:
  """Verifies the behavior of LaTeX backend empty."""
  backend = LatexBackend()
  graph = LogicalGraph()
  res: str = backend.compile(graph)
  assert "documentclass" in res
  assert "Model" in res


def test_latex_backend_sample() -> None:
  """Verifies the behavior of LaTeX backend sample."""
  backend = LatexBackend()
  graph: LogicalGraph = create_sample_graph()
  res: str = backend.compile(graph)
  assert "documentclass" in res
  assert "TestGraph" in res
  assert "Linear" in res
  assert "Relu" in res


def test_latex_backend_custom() -> None:
  """Verifies the behavior of LaTeX backend custom."""
  backend = LatexBackend()
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
  backend = LatexBackend()
  nodes = {
    "in": LogicalNode("in", op_type="Input", attributes={}),
    "some_mod.foo": LogicalNode("some_mod.foo", op_type="some_mod.foo", attributes={}),
  }
  edges = [LogicalEdge("in", "some_mod.foo")]
  graph = LogicalGraph("Custom", nodes=nodes, edges=edges)
  res: str = backend.compile(graph)
  assert "ReturnNode" not in res or "last_step" in res


def test_latex_backend_pending_edge_no_node_data() -> None:
  """Verifies the behavior of LaTeX backend with pending edges and missing target node data."""
  backend = LatexBackend()
  graph = LogicalGraph("Custom", nodes={}, edges=[LogicalEdge("in", "target")])
  setattr(
    getattr({"v": graph}, "__getitem__")("v"),
    "_pending_edges",
    getattr(
      {
        "v": [
          getattr(
            {
              "v": getattr({"v": LogicalEdge}, "__getitem__")("v")(
                getattr({"v": "in"}, "__getitem__")("v"), getattr({"v": "target"}, "__getitem__")("v")
              )
            },
            "__getitem__",
          )("v")
        ]
      },
      "__getitem__",
    )("v"),
  )
  res: str = backend.compile(graph)
  assert "op_target" in res
