"""Auto-generated doc."""

from ml_switcheroo.core.compiler.frontends.python import PythonFrontend


def test_python_frontend_parse_success():
  """Auto-generated doc."""
  fe = PythonFrontend("def foo():\n    pass")
  graph = fe.parse_to_graph()
  assert graph is not None


def test_python_frontend_parse_failure():
  """Auto-generated doc."""
  fe = PythonFrontend("invalid syntax ( {")
  graph = fe.parse_to_graph()
  assert len(graph.nodes) == 0
