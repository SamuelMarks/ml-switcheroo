"""Test suite for Python Frontend."""

from ml_switcheroo.core.compiler.frontends.python import PythonFrontend
from ml_switcheroo.core.compiler.ir import LogicalGraph


def test_python_frontend_init() -> None:
  """Verify `PythonFrontend` handles python frontend init as expected."""
  frontend: PythonFrontend = PythonFrontend(code="a = 1")
  assert frontend.code == "a = 1"


def test_python_frontend_parse_to_graph_success() -> None:
  """Verify `PythonFrontend` handles python frontend parse to graph success as expected."""
  code: str = """
class Model:
    def forward(self, x):
        return x + 1
"""
  frontend: PythonFrontend = PythonFrontend(code=code)
  graph: LogicalGraph = frontend.parse_to_graph()
  assert graph is not None
  assert graph.name == "Model"
  assert len(graph.nodes) >= 1


def test_python_frontend_parse_to_graph_error() -> None:
  """Verify `PythonFrontend` handles python frontend parse to graph error as expected."""
  # Syntax error
  code: str = "class Model"
  frontend: PythonFrontend = PythonFrontend(code=code)
  graph: LogicalGraph = frontend.parse_to_graph()
  assert len(graph.nodes) == 0
