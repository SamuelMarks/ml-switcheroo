"""Test suite for the Topk module."""

import pytest
import typing
import libcst as cst
from unittest.mock import MagicMock
from tests.conftest import TestRewriter as PivotRewriter
from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.plugins.topk import transform_topk


def rewrite_code(rewriter: PivotRewriter, code: str) -> str:
  """Rewrites code."""
  mod = cst.parse_module(code)
  return typing.cast(str, rewriter.convert(mod).code)


@pytest.fixture
def rewriter() -> PivotRewriter:
  """Provides a mock rewriter for testing."""
  hooks._HOOKS["topk_adapter"] = transform_topk
  hooks._PLUGINS_LOADED = True
  mgr = MagicMock()
  topk_def: dict[str, typing.Any] = {
    "variants": {
      "torch": {"api": "torch.topk"},
      "jax": {"api": "jax.lax.top_k", "requires_plugin": "topk_adapter"},
      "tensorflow": {"api": "tf.math.top_k", "requires_plugin": "topk_adapter"},
    }
  }

  def get_def_side_effect(n: str) -> typing.Optional[tuple[str, dict[str, typing.Any]]]:
    """Gets definition."""
    return ("TopK", topk_def) if "topk" in n else None

  def resolve_variant_side_effect(aid: str, fw: str) -> dict[str, typing.Any]:
    """Resolves variant."""
    return typing.cast(dict[str, typing.Any], topk_def["variants"]["jax"])

  mgr.get_definition.side_effect = get_def_side_effect
  mgr.resolve_variant.side_effect = resolve_variant_side_effect
  mgr.get_known_apis.return_value = {"TopK": topk_def}
  mgr.is_verified.return_value = True
  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(mgr, cfg)


def test_topk_rewrapping(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of topk rewrapping."""
  code: str = "res = torch.topk(x, 5)"
  res: str = rewrite_code(rewriter, code)
  assert "collections.namedtuple" in res
  assert '"TopK"' in res
  assert "jax.lax.top_k(x, 5)" in res
  assert "(*" in res


def test_topk_strip_unsupported(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of topk strip unsupported."""
  code: str = "res = torch.topk(x, 5, sorted=True)"
  res: str = rewrite_code(rewriter, code)
  assert "sorted" not in res
  assert "jax.lax.top_k(x, 5, )" in res or "jax.lax.top_k(x, 5)" in res


def test_topk_functional_call(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of topk functional call."""
  code: str = "res = torch.topk(t, k)"
  res: str = rewrite_code(rewriter, code)
  assert "jax.lax.top_k(t, k)" in res
