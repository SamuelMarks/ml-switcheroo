"""
Tests for Compiler Backend Protocol.

Verifies:
1. CompilerBackend abstract class enforcement.
2. Implementation of a concrete backend (NoOpBackend).
3. Interoperability with LogicalGraph IR.
"""

import pytest
from typing import Any
from ml_switcheroo.compiler.backend import CompilerBackend
from ml_switcheroo.compiler.ir import LogicalGraph, LogicalNode


class NoOpBackend(CompilerBackend):
  """
  A minimal backend that returns the graph node count as 'compiled' output.
  """

  def compile(self, graph: LogicalGraph) -> Any:
    return f"Compiled {len(graph.nodes)} nodes."


def test_backend_protocol_enforcement():
  """Verify that CompilerBackend cannot be instantiated directly."""
  with pytest.raises(TypeError):
    CompilerBackend()  # Abstract class


def test_noop_backend_compile():
  """Verify that a concrete backend correctly accepts LogicalGraph."""
  graph = LogicalGraph()
  graph.nodes = [
    LogicalNode(id="n1", kind="Input"),
    LogicalNode(id="n2", kind="Output"),
  ]

  backend = NoOpBackend()
  result = backend.compile(graph)

  assert result == "Compiled 2 nodes."


def test_backend_type_hints():
  """Verify that the abstract method signature matches protocol."""
  # This is implicit in the abc mechanism, but we verify method presence
  assert hasattr(CompilerBackend, "compile")
  assert CompilerBackend.compile.__isabstractmethod__
