"""Test suite for registry."""

import pytest

from ml_switcheroo.core.compiler.registry import GraphFrontend


def test_graph_frontend_parse_not_implemented():
  """Verify graph frontend parse not implemented behavior."""

  class DummyFrontend(GraphFrontend):
    """Test helper class DummyFrontend."""

    pass

  frontend = DummyFrontend()
  with pytest.raises(NotImplementedError):
    frontend.parse_to_graph("some code")
