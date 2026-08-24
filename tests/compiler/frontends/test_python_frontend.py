"""Docstring."""

from ml_switcheroo.core.compiler.frontends.python import PythonFrontend
from ml_switcheroo.core.compiler.ir import LogicalGraph


def test_python_frontend_parse_valid():
  """Docstring."""
  code = "import torch\nx = torch.add(a, b)"
  frontend = PythonFrontend(code)
  graph = frontend.parse_to_graph()
  assert isinstance(graph, LogicalGraph)
  # GraphExtractor should pull something
  assert len(graph.nodes) > 0


def test_python_frontend_parse_invalid():
  """Docstring."""
  code = "this is not valid python"
  frontend = PythonFrontend(code)
  graph = frontend.parse_to_graph()
  assert isinstance(graph, LogicalGraph)
  assert len(graph.nodes) == 0


def test_python_frontend_empty():
  """Docstring."""
  frontend = PythonFrontend("")
  graph = frontend.parse_to_graph()
  assert isinstance(graph, LogicalGraph)
