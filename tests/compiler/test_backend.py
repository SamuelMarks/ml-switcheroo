"""Test suite for the Backend module."""

import pytest
from typing import Any
from ml_switcheroo.core.compiler.backend import CompilerBackend
from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode


class NoOpBackend(CompilerBackend):
  """Test suite for the No Op Backend component."""

  def compile(self, graph: LogicalGraph) -> Any:
    """Compiles ."""
    return f"Compiled {len(graph.nodes)} nodes."


def test_backend_protocol_enforcement():
  """Verifies the behavior of backend protocol enforcement."""
  with pytest.raises(TypeError):
    CompilerBackend()


def test_noop_backend_compile():
  """Verifies the behavior of noop backend compile."""
  graph = LogicalGraph()
  graph.nodes = [LogicalNode(id="n1", kind="Input"), LogicalNode(id="n2", kind="Output")]
  backend = NoOpBackend()
  result = backend.compile(graph)
  assert result == "Compiled 2 nodes."


def test_backend_type_hints():
  """Verifies the behavior of backend type hints."""
  assert hasattr(CompilerBackend, "compile")
  assert CompilerBackend.compile.__isabstractmethod__
