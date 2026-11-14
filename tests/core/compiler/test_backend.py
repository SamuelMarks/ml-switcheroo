"""Test suite for the Backend module."""

import inspect

import pytest

from ml_switcheroo.core.compiler.backend import CompilerBackend
from ml_switcheroo.core.compiler.ir import IrCompilerBackend, LogicalGraph, LogicalNode


class NoOpBackend(CompilerBackend):
  """Stub `CompilerBackend` subclass implementing `compile` for backend tests."""

  def compile(self, graph: LogicalGraph) -> str:
    """Execute `compile` on `NoOpBackend` for backend tests.

    Args:
        graph: Logical computation graph instance (`LogicalGraph`).

    Returns:
        Resulting `str` from compile.
    """
    return f"Compiled {len(graph.nodes)} nodes."


def test_backend_protocol_enforcement() -> None:
  """Verify `TypeError` is raised during backend protocol enforcement."""
  incomplete_cls = type("_IncompleteBackend", (CompilerBackend,), {})
  with pytest.raises(TypeError):
    incomplete_cls()


def test_noop_backend_compile() -> None:
  """Verify `LogicalGraph` handles noop backend compile as expected."""
  graph = LogicalGraph(
    nodes={
      "n1": LogicalNode(id="n1", op_type="Input"),
      "n2": LogicalNode(id="n2", op_type="Output"),
    }
  )
  backend = NoOpBackend()
  assert isinstance(backend, IrCompilerBackend)
  result = backend.compile(graph)
  assert result == "Compiled 2 nodes."


def test_backend_type_hints() -> None:
  """Verify backend type hints produces the expected state."""
  assert inspect.isabstract(CompilerBackend)
  assert CompilerBackend.__abstractmethods__ == frozenset({"compile"})


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
