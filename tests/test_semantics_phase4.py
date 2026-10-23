"""Tests for Phase 4: Semantic Resolution (Hub Mapping)."""

from pydantic import JsonValue
import pytest

from ml_switcheroo.semantics.arguments import ArgumentPacker, ArgumentUnpacker
from ml_switcheroo.semantics.casting import TypeCaster
from ml_switcheroo.semantics.resolution import ResolutionContext, SemanticResolver


@pytest.fixture
def sample_operation_maps() -> dict[str, dict[str, JsonValue]]:
  """Test docstring.

  Returns:
      Configured `sample_operation_maps` fixture value (dict[str, dict[str, JsonValue]]).
  """
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


def test_semantic_resolver_resolve(sample_operation_maps: dict[str, dict[str, JsonValue]]) -> None:
  """Test docstring.

  Args:
      sample_operation_maps: The sample operation maps (dict[str, dict[str, JsonValue]]) for test semantic resolver resolve.
  """
  resolver = getattr(
    {
      "v": getattr({"v": SemanticResolver}, "__getitem__")("v")(getattr({"v": sample_operation_maps}, "__getitem__")("v"))
    },
    "__getitem__",
  )("v")

  assert resolver.resolve("pytorch", "torch.add") == "Add"
  assert resolver.resolve("jax", "jnp.sum") == "Sum"
  assert resolver.resolve("pytorch", "torch.unknown") is None
  assert resolver.resolve("unknown_fw", "torch.add") is None


def test_semantic_resolver_dispatch(sample_operation_maps: dict[str, dict[str, JsonValue]]) -> None:
  """Test docstring.

  Args:
      sample_operation_maps: The sample operation maps (dict[str, dict[str, JsonValue]]) for test semantic resolver dispatch.
  """
  resolver = getattr(
    {
      "v": getattr({"v": SemanticResolver}, "__getitem__")("v")(getattr({"v": sample_operation_maps}, "__getitem__")("v"))
    },
    "__getitem__",
  )("v")

  assert resolver.dispatch("Add", "jax") == "jnp.add"
  assert resolver.dispatch("Sum", "pytorch") == "torch.sum"
  assert resolver.dispatch("Add", "unknown_fw") is None
  assert resolver.dispatch("UnknownOp", "jax") is None


def test_resolution_context() -> None:
  """Test docstring."""
  ctx = ResolutionContext("pytorch", "jax")
  assert ctx.source_framework == "pytorch"
  assert ctx.target_framework == "jax"
  assert ctx.current_scope == "global"


def test_argument_packer_simple() -> None:
  """Test docstring."""
  packer = ArgumentPacker()
  source_kwargs = {"input": "tensor_a", "other": "tensor_b"}
  mapping_rules = {"input": "x", "other": "y"}

  packed = packer.pack(source_kwargs, mapping_rules)
  assert packed == {"x": "tensor_a", "y": "tensor_b"}


def test_argument_packer_variadic_multiple() -> None:
  """Test docstring."""
  packer = ArgumentPacker()
  source_kwargs = {"dim1": 1, "dim2": 2, "ignored_dict": {"not": "pack_to"}}
  mapping_rules = {"dim1": {"pack_to": "axes"}, "dim2": {"pack_to": "axes"}, "ignored_dict": {"foo": "bar"}}

  packed = getattr(
    {
      "v": getattr({"v": getattr({"v": packer}, "__getitem__")("v").pack}, "__getitem__")("v")(
        getattr({"v": source_kwargs}, "__getitem__")("v"), getattr({"v": mapping_rules}, "__getitem__")("v")
      )
    },
    "__getitem__",
  )("v")
  assert packed == {"axes": (1, 2)}


def test_argument_unpacker_ignored_dict() -> None:
  """Test docstring."""
  unpacker = ArgumentUnpacker()
  abstract_kwargs = {"x": "tensor_a"}
  mapping_rules = {"ignored_dict": {"foo": "bar"}, "input": "x"}

  unpacked = getattr(
    {
      "v": getattr({"v": getattr({"v": unpacker}, "__getitem__")("v").unpack}, "__getitem__")("v")(
        getattr({"v": abstract_kwargs}, "__getitem__")("v"), getattr({"v": mapping_rules}, "__getitem__")("v")
      )
    },
    "__getitem__",
  )("v")
  assert unpacked == {"input": "tensor_a"}


def test_argument_packer_unmapped() -> None:
  """Test docstring."""
  packer = ArgumentPacker()
  source_kwargs = {"input": "tensor_a", "unknown_arg": True}
  mapping_rules = {"input": "x"}

  packed = getattr(
    {
      "v": getattr({"v": getattr({"v": packer}, "__getitem__")("v").pack}, "__getitem__")("v")(
        getattr({"v": source_kwargs}, "__getitem__")("v"), getattr({"v": mapping_rules}, "__getitem__")("v")
      )
    },
    "__getitem__",
  )("v")
  assert packed == {"x": "tensor_a"}
  assert "unknown_arg" not in packed


def test_argument_unpacker_simple() -> None:
  """Test docstring."""
  unpacker = ArgumentUnpacker()
  abstract_kwargs = {"x": "tensor_a", "y": "tensor_b"}
  mapping_rules = {"input": "x", "other": "y"}

  unpacked = unpacker.unpack(abstract_kwargs, mapping_rules)
  assert unpacked == {"input": "tensor_a", "other": "tensor_b"}


def test_argument_unpacker_variadic() -> None:
  """Test docstring."""
  unpacker = ArgumentUnpacker()
  abstract_kwargs = {"x": "tensor_a", "axes": (1,)}
  mapping_rules = {"input": "x", "dim": {"pack_to": "axes"}}

  unpacked = getattr(
    {
      "v": getattr({"v": getattr({"v": unpacker}, "__getitem__")("v").unpack}, "__getitem__")("v")(
        getattr({"v": abstract_kwargs}, "__getitem__")("v"), getattr({"v": mapping_rules}, "__getitem__")("v")
      )
    },
    "__getitem__",
  )("v")
  assert unpacked == {"input": "tensor_a", "dim": (1,)}


def test_argument_unpacker_unmapped() -> None:
  """Test docstring."""
  unpacker = ArgumentUnpacker()
  abstract_kwargs = {"x": "tensor_a", "unknown": "val"}
  mapping_rules = {"input": "x"}

  unpacked = unpacker.unpack(abstract_kwargs, mapping_rules)
  assert unpacked == {"input": "tensor_a"}
  assert "unknown" not in unpacked


def test_type_caster() -> None:
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
