"""Module docstring."""

import pytest

from ml_switcheroo.core.compiler.registry import GraphFrontend


def test_graph_frontend_parse_not_implemented():
  """Docstring."""

  class DummyFrontend(GraphFrontend):
    """Docstring."""

    pass

  frontend = DummyFrontend()
  with pytest.raises(NotImplementedError):
    frontend.parse_to_graph("some code")
