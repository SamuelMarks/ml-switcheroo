"""Test suite for the Output Casting module."""

import libcst as cst
import pytest
from pydantic import JsonValue

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.enums import RuntimeTensorOrValue
from ml_switcheroo.semantics.manager import SemanticsManager
from tests.conftest import TestRewriter as PivotRewriter


class MockCastSemantics(SemanticsManager):
  """Test double `MockCastSemantics` used in output casting tests."""

  def __init__(self) -> None:
    """Initializes the MockCastSemantics instance."""
    self.data = {}
    self.import_data: dict[str, RuntimeTensorOrValue] = {}
    self.framework_configs = {}
    self._reverse_index = {}
    self._key_origins: dict[str, str] = {}
    self.test_templates = {}
    self._known_rng_methods: set[str] = set()
    self._inject("ArgMax", "torch.argmax", "jax.numpy.argmax", output_cast="jnp.int64")
    self._inject("Normalize", "torch.simple_op", "jax.op", output_cast="jnp.float32")

  def get_all_rng_methods(self) -> set[str]:
    """Mock implementation of get all rng methods.

    Returns:
        Resulting `set[str]` from get all rng methods.
    """
    return set()

  def get_framework_config(self, framework: str) -> dict[str, JsonValue]:
    """Mock implementation of get framework configuration.

    Args:
        framework: Framework identifier string.

    Returns:
        Resulting `dict[str, JsonValue]` from get framework config.
    """
    return {}

  def _inject(self, name: str, s_api: str, t_api: str, output_cast: str | None = None) -> None:
    """Execute `_inject` on `MockCastSemantics` for output casting tests.

    Args:
        name: Identifier or operation name string.
        s_api: Input `s_api` parameter value (`str`).
        t_api: Input `t_api` parameter value (`str`).
        output_cast: Input `output_cast` parameter value (`str | None`).
    """
    t_def: dict[str, JsonValue] = {"api": t_api}
    if output_cast:
      t_def["output_cast"] = output_cast
    variants: dict[str, JsonValue] = {"torch": {"api": s_api}, "jax": t_def}
    op_def: dict[str, JsonValue] = {
      "variants": variants,
      "std_args": ["x"],
    }
    self.data[name] = op_def
    self._reverse_index[s_api] = (name, op_def)

  def get_definition(self, name: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Mock implementation of get definition.

    Args:
        name: Identifier or operation name string.

    Returns:
        Resulting `tuple[str, dict[str, JsonValue]] | None` from get definition.
    """
    return self._reverse_index.get(name)

  def resolve_variant(self, abstract_id: str, fw: str) -> dict[str, JsonValue] | None:
    """Mock implementation of resolve variant.

    Args:
        abstract_id: Canonical abstract operation identifier (`str`).
        fw: Framework identifier string (e.g., torch, jax, mlx).

    Returns:
        Resulting `dict[str, JsonValue] | None` from resolve variant.
    """
    defn = self.data.get(abstract_id)
    if not isinstance(defn, dict):
      return None
    variants = defn.get("variants")
    if not isinstance(variants, dict):
      return None
    target = variants.get(fw)
    return target if isinstance(target, dict) else None


@pytest.fixture
def rewriter() -> PivotRewriter:
  """Create a configured `PivotRewriter` fixture for output casting tests.

  Returns:
      Configured `rewriter` fixture value (PivotRewriter).
  """
  semantics = MockCastSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=True)
  return PivotRewriter(semantics, config)


def rewrite(rewriter: PivotRewriter, code: str) -> str:
  """Execute `rewriter.convert` to support rewrite in output casting tests.

  Args:
      rewriter: Configured CST rewriter test helper instance.
      code: Python or DSL source code string to parse or transpile.

  Returns:
      Resulting `str` from rewrite.
  """
  tree = cst.parse_module(code)
  new_tree = rewriter.convert(tree)
  return str(new_tree.code)


def test_output_cast_injection(rewriter: PivotRewriter) -> None:
  """Verify `rewrite` emits `jax.numpy.argmax(x)` for output cast injection.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "y = torch.argmax(x)"
  result: str = rewrite(rewriter, code)
  assert "jax.numpy.argmax(x)" in result
  assert ".astype(jnp.int64)" in result


def test_output_cast_float_conversion(rewriter: PivotRewriter) -> None:
  """Verify `rewrite` emits `jax.op(x)` for output cast float conversion.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "z = torch.simple_op(x)"
  result: str = rewrite(rewriter, code)
  assert "jax.op(x)" in result
  assert ".astype(jnp.float32)" in result
