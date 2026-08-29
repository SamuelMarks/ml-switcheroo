"""Test suite for the Compiler Gap module."""

import typing


def test_compiler_backend_base() -> None:
  """Verifies the behavior of compiler backend base."""
  from ml_switcheroo.core.compiler.backend import CompilerBackend
  from ml_switcheroo.core.compiler.ir import LogicalGraph

  class DummyBackend(CompilerBackend):
    def compile(self, graph: LogicalGraph) -> str:
      """Mock implementation of compile."""
      return super().compile(graph)  # type: ignore

  backend = DummyBackend()
  import pytest

  with pytest.raises(NotImplementedError):
    backend.compile(None)  # type: ignore


def test_registry_gaps() -> None:
  """Verifies the behavior of registry gaps."""
  from ml_switcheroo.core.compiler.registry import get_backend_class, is_isa_source, is_isa_target

  cls: typing.Any = get_backend_class("unknown_target")
  assert cls.__name__ == "PythonBackend"
  assert is_isa_target("sass") is True
  assert is_isa_target("unknown") is False
  assert is_isa_source("rdna") is True
  assert is_isa_source("jax") is False


def test_sharding_extractor_gaps() -> None:
  """Docstring."""
  from ml_switcheroo.core.compiler.sharding_extractor import ShardingExtractionPass
  from ml_switcheroo.core.graph import LogicalEdge, LogicalGraph, LogicalNode

  graph = LogicalGraph(nodes=[LogicalNode("s", "with_sharding_constraint")], edges=[])
  ShardingExtractionPass().apply(graph)
  graph2 = LogicalGraph(nodes=[LogicalNode("s", "with_sharding_constraint")], edges=[LogicalEdge("missing", "s")])
  ShardingExtractionPass().apply(graph2)
