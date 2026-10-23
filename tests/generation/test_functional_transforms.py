"""Test suite for the Functional Transforms module."""

from ml_switcheroo.enums import RuntimeTensorOrValue

import typing
from pydantic import JsonValue

import pytest

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.engine import ASTEngine, ConversionResult
from ml_switcheroo.semantics.manager import SemanticsManager


class FunctionalSemantics(SemanticsManager):
  """Test helper class FunctionalSemantics."""

  def __init__(self) -> None:
    """Initializes the FunctionalSemantics instance."""
    self.data = {}
    self.framework_configs = {}
    self._reverse_index = {}
    self._key_origins: dict[str, str] = {}
    self._validation_status = {}
    self._known_rng_methods: set[str] = set()
    self._providers = {}
    self._source_registry = {}
    self.import_data: dict[str, RuntimeTensorOrValue] = {}
    self.data["vmap"] = {
      "std_args": ["func", "in_axes", "out_axes"],
      "variants": {
        "torch": {"api": "torch.vmap", "args": {"func": "func", "in_axes": "in_dims", "out_axes": "out_dims"}},
        "jax": {"api": "jax.vmap", "args": {"func": "fun"}},
      },
    }
    self.data["grad"] = {
      "std_args": ["func", "argnums"],
      "variants": {"torch": {"api": "torch.func.grad"}, "jax": {"api": "jax.grad", "args": {"func": "fun"}}},
    }
    self._reverse_index["torch.vmap"] = ("vmap", self.data["vmap"])
    self._reverse_index["jax.vmap"] = ("vmap", self.data["vmap"])
    self._reverse_index["torch.func.grad"] = ("grad", self.data["grad"])
    self._reverse_index["jax.grad"] = ("grad", self.data["grad"])

  def get_all_rng_methods(self) -> set[str]:
    """Gets all rng methods.

    Returns:
        Resulting `set[str]` from get all rng methods.
    """
    return set()

  def get_import_map(self, target_framework: str) -> dict[str, tuple[str, str | None, str | None]]:
    """Gets import map.

    Args:
        target_framework: Target framework identifier string.

    Returns:
        Resulting `dict[str, tuple[str, str | None, str | None]]` from get import map.
    """
    return {}

  def get_framework_config(self, framework: str) -> dict[str, JsonValue]:
    """Gets framework configuration.

    Returns:
        Configuration dictionary for the specified framework.
    """
    return {}

  def get_definition(self, name: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Mock get definition.

    Args:
        name: Identifier or operation name string.

    Returns:
        Tuple of abstract operation ID and definition metadata, or None.
    """
    return self._reverse_index.get(name)

  def resolve_variant(self, abstract_id: str, fw: str) -> dict[str, JsonValue] | None:
    """Mock resolve variant.

    Args:
        abstract_id: Canonical abstract operation identifier.
        fw: Framework identifier string (e.g., torch, jax, mlx).

    Returns:
        Target framework variant dictionary, or None if unsupported.
    """
    op = self.data.get(abstract_id, {})
    variants = op.get("variants")
    res = variants.get(fw) if isinstance(variants, dict) else None
    return res if isinstance(res, dict) else None

  def is_verified(self, _id: str) -> bool:
    """Mock is verified.

    Args:
        _id: Operation or node identifier string.

    Returns:
        True if the operation is verified, False otherwise.
    """
    return True


@pytest.fixture
def engine_factory() -> typing.Callable[[str, str], ASTEngine]:
  """Execute engine factory helper logic.

  Returns:
      Configured `engine_factory` fixture value (typing.Callable[[str, str], ASTEngine]).
  """
  semantics = FunctionalSemantics()

  def create(source: str, target: str) -> ASTEngine:
    """Creates .

    Args:
        source: Source framework identifier or input code string.
        target: Target framework or dialect identifier string.

    Returns:
        Resulting `ASTEngine` from create.
    """
    cfg = RuntimeConfig(source_framework=source, target_framework=target)
    return ASTEngine(semantics=semantics, config=cfg)

  return create


def test_torch_vmap_to_jax(engine_factory: typing.Callable[[str, str], ASTEngine]) -> None:
  """Verifies the behavior of PyTorch vmap to JAX.

  Args:
      engine_factory: Factory callable creating configured ASTEngine instances.
  """
  source_code: str = "v = torch.vmap(my_f, in_dims=(0, None))"
  engine = engine_factory("torch", "jax")
  result: ConversionResult = engine.run(source_code)
  assert result.success
  assert "jax.vmap" in result.code
  assert "in_axes=(0, None)" in result.code
  assert "in_dims" not in result.code


def test_jax_vmap_to_torch(engine_factory: typing.Callable[[str, str], ASTEngine]) -> None:
  """Verifies the behavior of JAX vmap to PyTorch.

  Args:
      engine_factory: Factory callable creating configured ASTEngine instances.
  """
  source_code: str = "v = jax.vmap(fun=f, in_axes=0)"
  engine = engine_factory("jax", "torch")
  result: ConversionResult = engine.run(source_code)
  assert result.success
  assert "torch.vmap" in result.code
  assert "in_dims=0" in result.code
  assert "func=f" in result.code


def test_grad_translation(engine_factory: typing.Callable[[str, str], ASTEngine]) -> None:
  """Verifies the behavior of grad translation.

  Args:
      engine_factory: Factory callable creating configured ASTEngine instances.
  """
  source_code: str = "g = torch.func.grad(predict)(params)"
  engine = engine_factory("torch", "jax")
  result: ConversionResult = engine.run(source_code)
  assert "jax.grad(predict)(params)" in result.code
