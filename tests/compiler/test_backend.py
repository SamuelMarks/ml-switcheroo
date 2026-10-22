"""Test suite for the Backend module."""

import pytest

from ml_switcheroo.core.compiler.backend import CompilerBackend
from ml_switcheroo.core.compiler.ir import IrCompilerBackend, LogicalGraph, LogicalNode


class NoOpBackend(CompilerBackend):
  """Test helper class NoOpBackend."""

  def compile(self, graph: LogicalGraph) -> str:
    """Compiles .

    Args:
        graph: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return f"Compiled {len(graph.nodes)} nodes."


def test_backend_protocol_enforcement() -> None:
  """Verifies the behavior of backend protocol enforcement."""
  cls = CompilerBackend
  with pytest.raises(TypeError):
    getattr({"v": cls}, "__getitem__")("v")()


def test_noop_backend_compile() -> None:
  """Verifies the behavior of noop backend compile."""
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
  """Verifies the behavior of backend type hints."""
  assert hasattr(CompilerBackend, "compile")
  assert getattr(CompilerBackend.compile, "__isabstractmethod__", False) is True


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
