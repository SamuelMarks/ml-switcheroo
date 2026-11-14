"""Test suite for the Topk module."""

from unittest.mock import MagicMock

import libcst as cst
import pytest
from pydantic import JsonValue

import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.plugins.topk import transform_topk
from tests.conftest import TestRewriter as PivotRewriter


def rewrite_code(rewriter: PivotRewriter, code: str) -> str:
  """Rewrites code.

  Args:
      code: Python or DSL source code string to parse or transpile.
      rewriter: Configured CST rewriter test helper instance.

  Returns:
      Resulting `str` from rewrite code.
  """
  mod = cst.parse_module(code)
  return str(rewriter.convert(mod).code)


@pytest.fixture
def rewriter() -> PivotRewriter:
  """Create a configured `PivotRewriter` fixture for topk tests.

  Returns:
      Configured `rewriter` fixture value (PivotRewriter).
  """
  hooks.register_hook("topk_adapter")(transform_topk)
  hooks.set_plugins_loaded(True)
  mgr = MagicMock()
  jax_variant: dict[str, JsonValue] = {"api": "jax.lax.top_k", "requires_plugin": "topk_adapter"}
  topk_def: dict[str, JsonValue] = {
    "variants": {
      "torch": {"api": "torch.topk"},
      "jax": jax_variant,
      "tensorflow": {"api": "tf.math.top_k", "requires_plugin": "topk_adapter"},
    }
  }

  def get_def_side_effect(n: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Gets definition.

    Args:
        n: Input `n` parameter value (`str`).

    Returns:
        Resulting `tuple[str, dict[str, JsonValue]] | None` from get def side effect.
    """
    return ("TopK", topk_def) if "topk" in n else None

  def resolve_variant_side_effect(aid: str, fw: str) -> dict[str, JsonValue]:
    """Resolves variant.

    Args:
        aid: Abstract operation identifier string.
        fw: Framework identifier string (e.g., torch, jax, mlx).

    Returns:
        Resulting `dict[str, JsonValue]` from resolve variant side effect.
    """
    return jax_variant

  mgr.get_definition.side_effect = get_def_side_effect
  mgr.resolve_variant.side_effect = resolve_variant_side_effect
  mgr.get_known_apis.return_value = {"TopK": topk_def}
  mgr.is_verified.return_value = True
  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(mgr, cfg)


def test_topk_rewrapping(rewriter: PivotRewriter) -> None:
  """Verify `rewrite_code` emits `collections.namedtuple` for topk rewrapping.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "res = torch.topk(x, 5)"
  res: str = rewrite_code(rewriter, code)
  assert "collections.namedtuple" in res
  assert '"TopK"' in res
  assert "jax.lax.top_k(x, 5)" in res
  assert "(*" in res


def test_topk_strip_unsupported(rewriter: PivotRewriter) -> None:
  """Verify `rewrite_code` handles topk strip unsupported as expected.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "res = torch.topk(x, 5, sorted=True)"
  res: str = rewrite_code(rewriter, code)
  assert "sorted" not in res
  assert "jax.lax.top_k(x, 5, )" in res or "jax.lax.top_k(x, 5)" in res


def test_topk_functional_call(rewriter: PivotRewriter) -> None:
  """Verify `rewrite_code` emits `jax.lax.top_k(t, k)` for topk functional call.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "res = torch.topk(t, k)"
  res: str = rewrite_code(rewriter, code)
  assert "jax.lax.top_k(t, k)" in res
