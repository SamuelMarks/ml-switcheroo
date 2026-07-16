"""Auto-generated doc."""


def test_compiler_backend_base():
  """Auto-generated doc."""
  from ml_switcheroo.core.compiler.backend import CompilerBackend

  class DummyBackend(CompilerBackend):
    """Auto-generated doc."""

    def compile(self, graph):
      """Auto-generated doc."""
      return super().compile(graph)

  backend = DummyBackend()
  import pytest

  with pytest.raises(NotImplementedError):
    backend.compile(None)


def test_registry_gaps():
  """Auto-generated doc."""
  from ml_switcheroo.core.compiler.registry import get_backend_class, is_isa_target, is_isa_source

  # 118: get_backend_class with unknown target falls back to python
  cls = get_backend_class("unknown_target")
  assert cls.__name__ == "PythonBackend"

  # 137: is_isa_target
  assert is_isa_target("sass") is True
  assert is_isa_target("unknown") is False

  # 151: is_isa_source
  assert is_isa_source("rdna") is True
  assert is_isa_source("jax") is False


def test_sharding_extractor_gaps():
  """Auto-generated doc."""
  from ml_switcheroo.core.compiler.sharding_extractor import ShardingExtractionPass
  from ml_switcheroo.core.graph import LogicalGraph, LogicalNode, LogicalEdge

  # Line 56: no source_id found (sharding node has no incoming edges)
  graph = LogicalGraph(nodes=[LogicalNode("s", "with_sharding_constraint")], edges=[])
  ShardingExtractionPass().apply(graph)

  # Line 66: source_id found but no source_node object (dangling edge)
  graph2 = LogicalGraph(nodes=[LogicalNode("s", "with_sharding_constraint")], edges=[LogicalEdge("missing", "s")])
  ShardingExtractionPass().apply(graph2)
