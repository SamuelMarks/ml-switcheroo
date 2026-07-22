"""Test suite for the Python Frontend module."""

from ml_switcheroo.core.compiler.frontends.python import PythonFrontend


def test_python_frontend_parse_success():
  """Verifies the behavior of python frontend parse successfully."""
  fe = PythonFrontend("def foo():\n    pass")
  graph = fe.parse_to_graph()
  assert graph is not None


def test_python_frontend_parse_failure():
  """Verifies the behavior of python frontend parse successfully handling failure."""
  fe = PythonFrontend("invalid syntax ( {")
  graph = fe.parse_to_graph()
  assert len(graph.nodes) == 0
