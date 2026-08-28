"""Test suite for the Output Casting module."""

import pytest
import typing
import libcst as cst
from tests.conftest import TestRewriter as PivotRewriter
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.config import RuntimeConfig


class MockCastSemantics(SemanticsManager):
  """Mock Cast Semantics class for testing purposes."""

  def __init__(self) -> None:
    """Initializes the MockCastSemantics instance."""
    self.data: dict[str, typing.Any] = {}
    self.import_data: dict[str, typing.Any] = {}
    self.framework_configs: dict[str, typing.Any] = {}
    self._reverse_index: dict[str, tuple[str, dict[str, typing.Any]]] = {}
    self._key_origins: dict[str, str] = {}
    self.test_templates: dict[str, typing.Any] = {}
    self._known_rng_methods: set[str] = set()
    self._inject("ArgMax", "torch.argmax", "jax.numpy.argmax", output_cast="jnp.int64")
    self._inject("Normalize", "torch.simple_op", "jax.op", output_cast="jnp.float32")

  def get_all_rng_methods(self) -> set[str]:
    """Mock implementation of get all rng methods."""
    return set()

  def get_framework_config(self, framework: str) -> dict[str, typing.Any]:
    """Mock implementation of get framework configuration."""
    return {}

  def _inject(self, name: str, s_api: str, t_api: str, output_cast: typing.Optional[str] = None) -> None:
    """Mock implementation of  inject."""
    t_def: dict[str, typing.Any] = {"api": t_api}
    if output_cast:
      t_def["output_cast"] = output_cast
    variants: dict[str, typing.Any] = {"torch": {"api": s_api}, "jax": t_def}
    self.data[name] = {"variants": variants, "std_args": ["x"]}
    self._reverse_index[s_api] = (name, self.data[name])

  def get_definition(self, name: str) -> typing.Optional[tuple[str, dict[str, typing.Any]]]:
    """Mock implementation of get definition."""
    return self._reverse_index.get(name)

  def resolve_variant(self, abstract_id: str, fw: str) -> typing.Any:
    """Mock implementation of resolve variant."""
    return self.data.get(abstract_id, {}).get("variants", {}).get(fw)


@pytest.fixture
def rewriter() -> PivotRewriter:
  """Provides a mock rewriter for testing."""
  semantics = MockCastSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=True)
  return PivotRewriter(semantics, config)


def rewrite(rewriter: PivotRewriter, code: str) -> str:
  """Rewrites ."""
  tree = cst.parse_module(code)
  new_tree: typing.Any = rewriter.convert(tree)
  return typing.cast(str, new_tree.code)


def test_output_cast_injection(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of output cast injection."""
  code: str = "y = torch.argmax(x)"
  result: str = rewrite(rewriter, code)
  assert "jax.numpy.argmax(x)" in result
  assert ".astype(jnp.int64)" in result


def test_output_cast_float_conversion(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of output cast float conversion."""
  code: str = "z = torch.simple_op(x)"
  result: str = rewrite(rewriter, code)
  assert "jax.op(x)" in result
  assert ".astype(jnp.float32)" in result
