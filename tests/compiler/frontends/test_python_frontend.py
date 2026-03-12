import pytest
from ml_switcheroo.compiler.frontends.python import PythonFrontend


def test_python_frontend_parse_success():
  fe = PythonFrontend("def foo():\n    pass")
  graph = fe.parse_to_graph()
  assert graph is not None


def test_python_frontend_parse_failure():
  fe = PythonFrontend("invalid syntax ( {")
  graph = fe.parse_to_graph()
  assert len(graph.nodes) == 0
