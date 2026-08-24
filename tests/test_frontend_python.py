"""Test suite for Python Frontend."""

from ml_switcheroo.core.compiler.frontends.python import PythonFrontend


def test_python_frontend_init():
  """Test element."""
  frontend = PythonFrontend(code="a = 1")
  assert frontend.code == "a = 1"


def test_python_frontend_parse_to_graph_success():
  """Test element."""
  code = """
class Model:
    def forward(self, x):
        return x + 1
"""
  frontend = PythonFrontend(code=code)
  graph = frontend.parse_to_graph()
  assert graph is not None


def test_python_frontend_parse_to_graph_error():
  """Test element."""
  # Syntax error
  code = "class Model"
  frontend = PythonFrontend(code=code)
  graph = frontend.parse_to_graph()
  assert len(graph.nodes) == 0
