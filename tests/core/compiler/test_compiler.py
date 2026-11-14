"""Test suite for the Compiler Gap module."""


def test_compiler_backend_base() -> None:
  """Verify `CompilerBackend.compile` raises `NotImplementedError` during compiler backend base."""
  from ml_switcheroo.core.compiler.backend import CompilerBackend
  from ml_switcheroo.core.compiler.ir import LogicalGraph

  class DummyBackend(CompilerBackend):
    """Stub `CompilerBackend` subclass implementing `compile` for compiler tests."""

    def compile(self, graph: LogicalGraph) -> str:
      """Mock implementation of compile.

      Args:
          graph: Logical computation graph instance (`LogicalGraph`).

      Returns:
          Resulting `str` from compile.
      """
      res: str = CompilerBackend.compile(self, graph)
      return res

  backend = DummyBackend()
  import pytest

  with pytest.raises(NotImplementedError):
    backend.compile(LogicalGraph())


def test_registry_gaps() -> None:
  """Verify `get_backend_class` handles registry gaps as expected."""
  from ml_switcheroo.core.compiler.registry import get_backend_class, is_isa_source, is_isa_target

  cls = get_backend_class("unknown_target")
  assert cls is not None
  assert cls.__name__ == "PythonBackend"
  assert is_isa_target("nvidia_sass") is True
  assert is_isa_target("unknown") is False
  assert is_isa_source("rdna") is True
  assert is_isa_source("jax") is False


def test_sharding_extractor_gaps() -> None:
  """Verify `LogicalGraph` emits `s` for sharding extractor gaps."""
  from ml_switcheroo.core.compiler.sharding_extractor import ShardingExtractionPass
  from ml_switcheroo.core.graph import LogicalEdge, LogicalGraph, LogicalNode

  graph = LogicalGraph(nodes={n.id: n for n in [LogicalNode("s", "with_sharding_constraint")]}, edges=[])
  ShardingExtractionPass().apply(graph)
  graph2 = LogicalGraph(
    nodes={n.id: n for n in [LogicalNode("s", "with_sharding_constraint")]}, edges=[LogicalEdge("missing", "s")]
  )
  _res = ShardingExtractionPass().apply(graph2)
  assert "s" in _res.nodes
  assert len(_res.edges) == 1
