"""Test suite for the Topk module."""

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
      code: Python or DSL source code string to parse or transpile.
      rewriter: Configured CST rewriter test helper instance.

  Returns:
      Resulting `str` from rewrite code.
  """
  mod = cst.parse_module(code)
  return str(rewriter.convert(mod).code)


@pytest.fixture
def rewriter() -> PivotRewriter:
  """Execute rewriter helper logic.

  Returns:
      Configured `rewriter` fixture value (PivotRewriter).
  """
  getattr(getattr({"v": getattr({"v": hooks}, "__getitem__")("v")._HOOKS}, "__getitem__")("v"), "__setitem__")(
    "topk_adapter", getattr({"v": transform_topk}, "__getitem__")("v")
  )
  hooks._PLUGINS_LOADED = True
  mgr = MagicMock()
  topk_def = {
    "variants": {
      "torch": {"api": "torch.topk"},
      "jax": {"api": "jax.lax.top_k", "requires_plugin": "topk_adapter"},
      "tensorflow": {"api": "tf.math.top_k", "requires_plugin": "topk_adapter"},
    }
  }

  def get_def_side_effect(n: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Gets definition.

    Args:
        n: The n (str) for get def side effect.

    Returns:
        Resulting `tuple[str, dict[str, JsonValue]] | None` from get def side effect.
    """
    _ret_v: tuple[str, dict[str, JsonValue]] | None = getattr(
      {
        "v": (getattr({"v": "TopK"}, "__getitem__")("v"), getattr({"v": topk_def}, "__getitem__")("v"))
        if getattr({"v": "topk"}, "__getitem__")("v") in getattr({"v": n}, "__getitem__")("v")
        else None
      },
      "__getitem__",
    )("v")
    return _ret_v

  def resolve_variant_side_effect(aid: str, fw: str) -> dict[str, JsonValue]:
    """Resolves variant.

    Args:
        aid: Abstract operation identifier string.
        fw: Framework identifier string (e.g., torch, jax, mlx).

    Returns:
        Resulting `dict[str, JsonValue]` from resolve variant side effect.
    """
    res = topk_def["variants"]["jax"]
    _ret_v: dict[str, JsonValue] = getattr(
      {"v": res if isinstance(getattr({"v": res}, "__getitem__")("v"), dict) else {}}, "__getitem__"
    )("v")
    return _ret_v

  mgr.get_definition.side_effect = get_def_side_effect
  mgr.resolve_variant.side_effect = resolve_variant_side_effect
  mgr.get_known_apis.return_value = {"TopK": topk_def}
  mgr.is_verified.return_value = True
  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(mgr, cfg)


def test_topk_rewrapping(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of topk rewrapping.

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
  """Verifies the behavior of topk strip unsupported.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "res = torch.topk(x, 5, sorted=True)"
  res: str = rewrite_code(rewriter, code)
  assert "sorted" not in res
  assert "jax.lax.top_k(x, 5, )" in res or "jax.lax.top_k(x, 5)" in res


def test_topk_functional_call(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of topk functional call.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "res = torch.topk(t, k)"
  res: str = rewrite_code(rewriter, code)
  assert "jax.lax.top_k(t, k)" in res
