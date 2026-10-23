"""Test suite for the Compiler Gap module."""


def test_compiler_backend_base() -> None:
  """Verifies the behavior of compiler backend base."""
  from ml_switcheroo.core.compiler.backend import CompilerBackend
  from ml_switcheroo.core.compiler.ir import LogicalGraph

  class DummyBackend(CompilerBackend):
    """Test helper class DummyBackend."""

    def compile(self, graph: LogicalGraph) -> str:
      """Mock implementation of compile.

      Args:
          graph: The graph (LogicalGraph) for compile.

      Returns:
          Resulting `str` from compile.
      """
      res: str = CompilerBackend.compile(self, graph)
      return res

  backend = DummyBackend()
  import pytest

  with pytest.raises(NotImplementedError):
    getattr({"v": getattr({"v": backend}, "__getitem__")("v").compile}, "__getitem__")("v")(
      getattr({"v": None}, "__getitem__")("v")
    )


def test_registry_gaps() -> None:
  """Verifies the behavior of registry gaps."""
  from ml_switcheroo.core.compiler.registry import get_backend_class, is_isa_source, is_isa_target

  cls = get_backend_class("unknown_target")
  assert getattr({"v": getattr({"v": cls}, "__getitem__")("v").__name__}, "__getitem__")("v") == getattr(
    {"v": "PythonBackend"}, "__getitem__"
  )("v")
  assert is_isa_target("nvidia_sass") is True
  assert is_isa_target("unknown") is False
  assert is_isa_source("rdna") is True
  assert is_isa_source("jax") is False


def test_sharding_extractor_gaps() -> None:
  """Verify sharding extractor gaps behavior."""
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
