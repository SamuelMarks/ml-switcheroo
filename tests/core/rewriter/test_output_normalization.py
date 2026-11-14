"""Test suite for the Output Normalization module."""

import libcst as cst
import pytest
from pydantic import JsonValue

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.enums import RuntimeTensorOrValue
from ml_switcheroo.semantics.manager import SemanticsManager
from tests.conftest import TestRewriter


class MockOutputSemantics(SemanticsManager):
  """Test double `MockOutputSemantics` used in output normalization tests."""

  def __init__(self) -> None:
    """Initializes the MockOutputSemantics instance."""
    self.data = {}
    self._reverse_index = {}
    self._key_origins: dict[str, str] = {}
    self.import_data: dict[str, RuntimeTensorOrValue] = {}
    self.framework_configs = {}
    self._inject("split_vals", "torch.split", "jax.numpy.split", select_index=0)

  def get_framework_config(self, framework: str) -> dict[str, JsonValue]:
    """Mock implementation of get framework configuration.

    Args:
        framework: Framework identifier string.

    Returns:
        Resulting `dict[str, JsonValue]` from get framework config.
    """
    return self.framework_configs.get(framework, {})

  def _inject(self, name: str, s_api: str, t_api: str, select_index: int | None = None) -> None:
    """Execute `_inject` on `MockOutputSemantics` for output normalization tests.

    Args:
        name: Identifier or operation name string.
        s_api: Input `s_api` parameter value (`str`).
        t_api: Input `t_api` parameter value (`str`).
        select_index: Input `select_index` parameter value (`int | None`).
    """
    target_var: dict[str, JsonValue] = {"api": t_api}
    if select_index is not None:
      target_var["output_select_index"] = select_index
    variants: dict[str, JsonValue] = {"torch": {"api": s_api}, "jax": target_var}
    op_def: dict[str, JsonValue] = {
      "variants": variants,
      "std_args": ["x", "y"],
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

  def get_import_map(self, target_framework: str) -> dict[str, tuple[str, str | None, str | None]]:
    """Mock implementation of get import map.

    Args:
        target_framework: Input `target_framework` parameter value (`str`).

    Returns:
        Resulting `dict[str, tuple[str, str | None, str | None]]` from get import map.
    """
    return {}


@pytest.fixture
def rewriter() -> TestRewriter:
  """Create a configured `TestRewriter` fixture for output normalization tests.

  Returns:
      Configured `rewriter` fixture value (TestRewriter).
  """
  semantics = MockOutputSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=True)
  return TestRewriter(semantics, config)


def rewrite(rewriter: TestRewriter, code: str) -> str:
  """Execute `rewriter.convert` to support rewrite in output normalization tests.

  Args:
      rewriter: Configured CST rewriter test helper instance.
      code: Python or DSL source code string to parse or transpile.

  Returns:
      Resulting `str` from rewrite.
  """
  tree = cst.parse_module(code)
  new_tree = rewriter.convert(tree)
  return str(new_tree.code)


def test_structured_index_wrapping(rewriter: TestRewriter) -> None:
  """Verify `rewrite` emits `jax.numpy.split(x)[0]` for structured index wrapping.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "res = torch.split(x)"
  result: str = rewrite(rewriter, code)
  assert "jax.numpy.split(x)[0]" in result
  assert "lambda" not in result
