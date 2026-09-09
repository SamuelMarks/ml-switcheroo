"""Tests for Phase 4: Semantic Resolution (Hub Mapping)."""

import pytest

from ml_switcheroo.semantics.resolution import SemanticResolver, ResolutionContext
from ml_switcheroo.semantics.arguments import ArgumentPacker, ArgumentUnpacker
from ml_switcheroo.semantics.casting import TypeCaster


@pytest.fixture
def sample_operation_maps():
  """Test docstring."""
  return {
    "Add": {
      "variants": {
        "pytorch": {"api": "torch.add", "args": {"input": "x", "other": "y"}},
        "jax": {"api": "jnp.add", "args": {"x1": "x", "x2": "y"}},
      }
    },
    "Sum": {
      "variants": {
        "pytorch": {"api": "torch.sum", "args": {"input": "x", "dim": {"pack_to": "axes"}}},
        "jax": {"api": "jnp.sum", "args": {"a": "x", "axis": {"pack_to": "axes"}}},
      }
    },
  }


def test_semantic_resolver_resolve(sample_operation_maps):
  """Test docstring."""
  resolver = SemanticResolver(sample_operation_maps)

  assert resolver.resolve("pytorch", "torch.add") == "Add"
  assert resolver.resolve("jax", "jnp.sum") == "Sum"
  assert resolver.resolve("pytorch", "torch.unknown") is None
  assert resolver.resolve("unknown_fw", "torch.add") is None


def test_semantic_resolver_dispatch(sample_operation_maps):
  """Test docstring."""
  resolver = SemanticResolver(sample_operation_maps)

  assert resolver.dispatch("Add", "jax") == "jnp.add"
  assert resolver.dispatch("Sum", "pytorch") == "torch.sum"
  assert resolver.dispatch("Add", "unknown_fw") is None
  assert resolver.dispatch("UnknownOp", "jax") is None


def test_resolution_context():
  """Test docstring."""
  ctx = ResolutionContext("pytorch", "jax")
  assert ctx.source_framework == "pytorch"
  assert ctx.target_framework == "jax"
  assert ctx.current_scope == "global"


def test_argument_packer_simple():
  """Test docstring."""
  packer = ArgumentPacker()
  source_kwargs = {"input": "tensor_a", "other": "tensor_b"}
  mapping_rules = {"input": "x", "other": "y"}

  packed = packer.pack(source_kwargs, mapping_rules)
  assert packed == {"x": "tensor_a", "y": "tensor_b"}


def test_argument_packer_variadic_multiple():
  """Test docstring."""
  packer = ArgumentPacker()
  source_kwargs = {"dim1": 1, "dim2": 2, "ignored_dict": {"not": "pack_to"}}
  mapping_rules = {"dim1": {"pack_to": "axes"}, "dim2": {"pack_to": "axes"}, "ignored_dict": {"foo": "bar"}}

  packed = packer.pack(source_kwargs, mapping_rules)
  assert packed == {"axes": (1, 2)}


def test_argument_unpacker_ignored_dict():
  """Test docstring."""
  unpacker = ArgumentUnpacker()
  abstract_kwargs = {"x": "tensor_a"}
  mapping_rules = {"ignored_dict": {"foo": "bar"}, "input": "x"}

  unpacked = unpacker.unpack(abstract_kwargs, mapping_rules)
  assert unpacked == {"input": "tensor_a"}


def test_argument_packer_unmapped():
  """Test docstring."""
  packer = ArgumentPacker()
  source_kwargs = {"input": "tensor_a", "unknown_arg": True}
  mapping_rules = {"input": "x"}

  packed = packer.pack(source_kwargs, mapping_rules)
  assert packed == {"x": "tensor_a"}
  assert "unknown_arg" not in packed


def test_argument_unpacker_simple():
  """Test docstring."""
  unpacker = ArgumentUnpacker()
  abstract_kwargs = {"x": "tensor_a", "y": "tensor_b"}
  mapping_rules = {"input": "x", "other": "y"}

  unpacked = unpacker.unpack(abstract_kwargs, mapping_rules)
  assert unpacked == {"input": "tensor_a", "other": "tensor_b"}


def test_argument_unpacker_variadic():
  """Test docstring."""
  unpacker = ArgumentUnpacker()
  abstract_kwargs = {"x": "tensor_a", "axes": (1,)}
  mapping_rules = {"input": "x", "dim": {"pack_to": "axes"}}

  unpacked = unpacker.unpack(abstract_kwargs, mapping_rules)
  assert unpacked == {"input": "tensor_a", "dim": (1,)}


def test_argument_unpacker_unmapped():
  """Test docstring."""
  unpacker = ArgumentUnpacker()
  abstract_kwargs = {"x": "tensor_a", "unknown": "val"}
  mapping_rules = {"input": "x"}

  unpacked = unpacker.unpack(abstract_kwargs, mapping_rules)
  assert unpacked == {"input": "tensor_a"}
  assert "unknown" not in unpacked


def test_type_caster():
  """Test docstring."""
  caster = TypeCaster()

  # Direct mappings
  assert caster.cast("torch.float32", "pytorch", "jax") == "jnp.float32"
  assert caster.cast("jnp.float64", "jax", "pytorch") == "torch.float64"

  # Unknown source type falls back
  assert caster.cast("unknown_type", "pytorch", "jax") == "unknown_type"

  # Unknown target framework falls back to normalized
  assert caster.cast("torch.float16", "pytorch", "unknown_fw") == "float16"

  # Unknown source framework but known target
  assert caster.cast("float32", "unknown_fw", "pytorch") == "torch.float32"
