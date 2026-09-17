"""Test suite for the Backend module."""

import pytest

from ml_switcheroo.core.compiler.backend import CompilerBackend
from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode


class NoOpBackend(CompilerBackend):
  """Docstring."""

  def compile(self, graph: LogicalGraph) -> str:
    """Compiles ."""
    return f"Compiled {len(graph.nodes)} nodes."


def test_backend_protocol_enforcement() -> None:
  """Verifies the behavior of backend protocol enforcement."""
  with pytest.raises(TypeError):
    CompilerBackend()  # type: ignore


def test_noop_backend_compile() -> None:
  """Verifies the behavior of noop backend compile."""
  graph = LogicalGraph()
  graph.nodes = [LogicalNode(id="n1", op_type="Input"), LogicalNode(id="n2", op_type="Output")]
  backend = NoOpBackend()
  result: str = backend.compile(graph)
  assert result == "Compiled 2 nodes."


def test_backend_type_hints() -> None:
  """Verifies the behavior of backend type hints."""
  assert hasattr(CompilerBackend, "compile")
  assert CompilerBackend.compile.__isabstractmethod__  # type: ignore


def test_compiler_backend_super_call() -> None:
  """Verifies calling super().compile raises NotImplementedError."""

  class ConcreteBackend(CompilerBackend):
    """Concrete backend for testing base method."""

    def compile(self, graph: LogicalGraph) -> str:
      """Compile method.

      Args:
          graph: Input logical graph.

      Returns:
          Compiled string.
      """
      return ""

  backend = ConcreteBackend()
  with pytest.raises(NotImplementedError):
    CompilerBackend.compile(backend, LogicalGraph())
