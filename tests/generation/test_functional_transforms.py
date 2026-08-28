"""Test suite for the Functional Transforms module."""

import pytest
import typing
from ml_switcheroo.core.engine import ASTEngine, ConversionResult
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager


class FunctionalSemantics(SemanticsManager):
  """Test suite for the Functional Semantics component."""

  def __init__(self) -> None:
    """Initializes the FunctionalSemantics instance."""
    self.data: dict[str, typing.Any] = {}
    self.framework_configs: dict[str, typing.Any] = {}
    self._reverse_index: dict[str, tuple[str, dict[str, typing.Any]]] = {}
    self._key_origins: dict[str, str] = {}
    self._validation_status: dict[str, typing.Any] = {}
    self._known_rng_methods: set[str] = set()
    self._providers: dict[str, typing.Any] = {}
    self._source_registry: dict[str, typing.Any] = {}
    self.import_data: dict[str, typing.Any] = {}
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
    """Gets all rng methods."""
    return set()

  def get_import_map(self, target_fw: str) -> dict[str, tuple[str, typing.Optional[str], typing.Optional[str]]]:
    """Gets import map."""
    return {}

  def get_framework_config(self, framework: str) -> dict[str, typing.Any]:
    """Gets framework configuration."""
    return {}

  def get_definition(self, name: str) -> typing.Optional[tuple[str, dict[str, typing.Any]]]:
    """Mock get definition."""
    return self._reverse_index.get(name)

  def resolve_variant(self, abstract_id: str, fw: str) -> typing.Any:
    """Mock resolve variant."""
    return self.data.get(abstract_id, {}).get("variants", {}).get(fw)

  def is_verified(self, _id: str) -> bool:
    """Mock is verified."""
    return True


@pytest.fixture
def engine_factory() -> typing.Callable[[str, str], ASTEngine]:
  """Provides a mock engine factory for testing."""
  semantics = FunctionalSemantics()

  def create(source: str, target: str) -> ASTEngine:
    """Creates ."""
    cfg = RuntimeConfig(source_framework=source, target_framework=target)
    return ASTEngine(semantics=semantics, config=cfg)

  return create


def test_torch_vmap_to_jax(engine_factory: typing.Callable[[str, str], ASTEngine]) -> None:
  """Verifies the behavior of PyTorch vmap to JAX."""
  source_code: str = "v = torch.vmap(my_f, in_dims=(0, None))"
  engine = engine_factory("torch", "jax")
  result: ConversionResult = engine.run(source_code)
  assert result.success
  assert "jax.vmap" in result.code
  assert "in_axes=(0, None)" in result.code
  assert "in_dims" not in result.code


def test_jax_vmap_to_torch(engine_factory: typing.Callable[[str, str], ASTEngine]) -> None:
  """Verifies the behavior of JAX vmap to PyTorch."""
  source_code: str = "v = jax.vmap(fun=f, in_axes=0)"
  engine = engine_factory("jax", "torch")
  result: ConversionResult = engine.run(source_code)
  assert result.success
  assert "torch.vmap" in result.code
  assert "in_dims=0" in result.code
  assert "func=f" in result.code


def test_grad_translation(engine_factory: typing.Callable[[str, str], ASTEngine]) -> None:
  """Verifies the behavior of grad translation."""
  source_code: str = "g = torch.func.grad(predict)(params)"
  engine = engine_factory("torch", "jax")
  result: ConversionResult = engine.run(source_code)
  assert "jax.grad(predict)(params)" in result.code
