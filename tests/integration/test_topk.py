"""Test suite for the Topk module."""

import typing
from pydantic import JsonValue
from unittest.mock import MagicMock

import libcst as cst
import pytest

import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.plugins.topk import transform_topk
from tests.conftest import TestRewriter as PivotRewriter


def rewrite_code(rewriter: PivotRewriter, code: str) -> str:
  """Rewrites code.

  Args:
      code: Test argument or fixture.
      rewriter: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  mod = cst.parse_module(code)
  return str(rewriter.convert(mod).code)


@pytest.fixture
def rewriter() -> PivotRewriter:
  """Execute rewriter helper logic.

  Returns:
      Computed test result or fixture value.
  """
  hooks._HOOKS["topk_adapter"] = transform_topk  # type: ignore[assignment]
  hooks._PLUGINS_LOADED = True
  mgr = MagicMock()
  topk_def = {
    "variants": {
      "torch": {"api": "torch.topk"},
      "jax": {"api": "jax.lax.top_k", "requires_plugin": "topk_adapter"},
      "tensorflow": {"api": "tf.math.top_k", "requires_plugin": "topk_adapter"},
    }
  }

  def get_def_side_effect(n: str) -> typing.Optional[tuple[str, dict[str, JsonValue]]]:
    """Gets definition.

    Args:
        n: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return ("TopK", topk_def) if "topk" in n else None  # type: ignore[return-value]

  def resolve_variant_side_effect(aid: str, fw: str) -> dict[str, JsonValue]:
    """Resolves variant.

    Args:
        aid: Test argument or fixture.
        fw: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    res = topk_def["variants"]["jax"]
    return res if isinstance(res, dict) else {}  # type: ignore[return-value, redundant-expr]

  mgr.get_definition.side_effect = get_def_side_effect
  mgr.resolve_variant.side_effect = resolve_variant_side_effect
  mgr.get_known_apis.return_value = {"TopK": topk_def}
  mgr.is_verified.return_value = True
  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(mgr, cfg)


def test_topk_rewrapping(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of topk rewrapping.

  Args:
      rewriter: Test argument or fixture.
  """
  code: str = "res = torch.topk(x, 5)"
  res: str = rewrite_code(rewriter, code)
  assert "collections.namedtuple" in res
  assert '"TopK"' in res
  assert "jax.lax.top_k(x, 5)" in res
  assert "(*" in res


def test_topk_strip_unsupported(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of topk strip unsupported.

  Args:
      rewriter: Test argument or fixture.
  """
  code: str = "res = torch.topk(x, 5, sorted=True)"
  res: str = rewrite_code(rewriter, code)
  assert "sorted" not in res
  assert "jax.lax.top_k(x, 5, )" in res or "jax.lax.top_k(x, 5)" in res


def test_topk_functional_call(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of topk functional call.

  Args:
      rewriter: Test argument or fixture.
  """
  code: str = "res = torch.topk(t, k)"
  res: str = rewrite_code(rewriter, code)
  assert "jax.lax.top_k(t, k)" in res
