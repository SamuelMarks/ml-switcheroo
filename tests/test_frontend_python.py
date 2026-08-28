"""Test suite for Python Frontend."""

from ml_switcheroo.core.compiler.frontends.python import PythonFrontend
from ml_switcheroo.core.compiler.ir import LogicalGraph


def test_python_frontend_init() -> None:
  """Test element."""
  frontend: PythonFrontend = PythonFrontend(code="a = 1")
  assert frontend.code == "a = 1"


def test_python_frontend_parse_to_graph_success() -> None:
  """Test element."""
  code: str = """
class Model:
    def forward(self, x):
        return x + 1
"""
  frontend: PythonFrontend = PythonFrontend(code=code)
  graph: LogicalGraph = frontend.parse_to_graph()
  assert graph is not None


def test_python_frontend_parse_to_graph_error() -> None:
  """Test element."""
  # Syntax error
  code: str = "class Model"
  frontend: PythonFrontend = PythonFrontend(code=code)
  graph: LogicalGraph = frontend.parse_to_graph()
  assert len(graph.nodes) == 0
