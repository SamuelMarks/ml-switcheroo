"""Test suite for the Compiler Gap module."""


def test_compiler_backend_base():
  """Verifies the behavior of compiler backend base."""
  from ml_switcheroo.core.compiler.backend import CompilerBackend

  class DummyBackend(CompilerBackend):
    """Dummy Backend class for testing purposes."""

    def compile(self, graph):
      """Mock implementation of compile."""
      return super().compile(graph)

  backend = DummyBackend()
  import pytest

  with pytest.raises(NotImplementedError):
    backend.compile(None)


def test_registry_gaps():
  """Verifies the behavior of registry gaps."""
  from ml_switcheroo.core.compiler.registry import get_backend_class, is_isa_target, is_isa_source

  cls = get_backend_class("unknown_target")
  assert cls.__name__ == "PythonBackend"
  assert is_isa_target("sass") is True
  assert is_isa_target("unknown") is False
  assert is_isa_source("rdna") is True
  assert is_isa_source("jax") is False


def test_sharding_extractor_gaps():
  """Verifies the behavior of sharding extractor gaps."""
  from ml_switcheroo.core.compiler.sharding_extractor import ShardingExtractionPass
  from ml_switcheroo.core.graph import LogicalGraph, LogicalNode, LogicalEdge

  graph = LogicalGraph(nodes=[LogicalNode("s", "with_sharding_constraint")], edges=[])
  ShardingExtractionPass().apply(graph)
  graph2 = LogicalGraph(nodes=[LogicalNode("s", "with_sharding_constraint")], edges=[LogicalEdge("missing", "s")])
  ShardingExtractionPass().apply(graph2)
