"""Test suite for the Functional Transforms module."""

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
    self.import_data: dict[str, typing.Union[typing.Hashable, typing.Sized]] = {}
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
        Computed test result or fixture value.
    """
    return set()

  def get_import_map(self, target_fw: str) -> dict[str, tuple[str, typing.Optional[str], typing.Optional[str]]]:
    """Gets import map.

    Args:
        target_fw: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return {}

  def get_framework_config(self, framework: str) -> dict[str, JsonValue]:
    """Gets framework configuration.

    Returns:
        Computed test result or fixture value.
    """
    return {}

  def get_definition(self, name: str) -> typing.Optional[tuple[str, dict[str, JsonValue]]]:
    """Mock get definition.

    Args:
        name: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self._reverse_index.get(name)

  def resolve_variant(self, abstract_id: str, fw: str) -> typing.Optional[dict[str, JsonValue]]:
    """Mock resolve variant.

    Args:
        abstract_id: Test argument or fixture.
        fw: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self.data.get(abstract_id, {}).get("variants", {}).get(fw)  # type: ignore[union-attr, return-value]

  def is_verified(self, _id: str) -> bool:
    """Mock is verified.

    Args:
        _id: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return True


@pytest.fixture
def engine_factory() -> typing.Callable[[str, str], ASTEngine]:
  """Execute engine factory helper logic.

  Returns:
      Computed test result or fixture value.
  """
  semantics = FunctionalSemantics()

  def create(source: str, target: str) -> ASTEngine:
    """Creates .

    Args:
        source: Test argument or fixture.
        target: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    cfg = RuntimeConfig(source_framework=source, target_framework=target)
    return ASTEngine(semantics=semantics, config=cfg)

  return create


def test_torch_vmap_to_jax(engine_factory: typing.Callable[[str, str], ASTEngine]) -> None:
  """Verifies the behavior of PyTorch vmap to JAX.

  Args:
      engine_factory: Test argument or fixture.
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
      engine_factory: Test argument or fixture.
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
      engine_factory: Test argument or fixture.
  """
  source_code: str = "g = torch.func.grad(predict)(params)"
  engine = engine_factory("torch", "jax")
  result: ConversionResult = engine.run(source_code)
  assert "jax.grad(predict)(params)" in result.code
