"""Docstring."""

from ml_switcheroo.core.compiler.frontends.python import PythonFrontend
from ml_switcheroo.core.compiler.ir import LogicalGraph


def test_python_frontend_parse_valid() -> None:
  """Docstring."""
  code: str = "import torch\nx = torch.add(a, b)"
  frontend = PythonFrontend(code)
  graph: LogicalGraph = frontend.parse_to_graph()
  assert isinstance(graph, LogicalGraph)
  # GraphExtractor should pull something
  assert len(graph.nodes) > 0


def test_python_frontend_parse_invalid() -> None:
  """Docstring."""
  code: str = "this is not valid python"
  frontend = PythonFrontend(code)
  graph: LogicalGraph = frontend.parse_to_graph()
  assert isinstance(graph, LogicalGraph)
  assert len(graph.nodes) == 0


def test_python_frontend_empty() -> None:
  """Docstring."""
  frontend = PythonFrontend("")
  graph: LogicalGraph = frontend.parse_to_graph()
  assert isinstance(graph, LogicalGraph)
