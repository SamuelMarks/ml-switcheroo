"""Test suite for plugin coverage (`in_top_k_plugin`)."""

from unittest import mock

import libcst as cst

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.in_top_k_plugin import in_top_k_plugin
from ml_switcheroo.semantics.schema import PluginTraits


def test_in_top_k_plugin_fewer_than_three_args() -> None:
  """Verify `in_top_k_plugin` returns the original node when fewer than 3 arguments are given."""
  node: cst.Call = cst.Call(func=cst.Name("in_top_k"), args=[])
  ctx: mock.MagicMock = mock.MagicMock()
  result: cst.CSTNode = in_top_k_plugin(node, ctx)
  assert result is node


def test_in_top_k_plugin_with_lookup_api() -> None:
  """Verify `in_top_k_plugin` rewrites 3-argument call when `lookup_api('TopK')` succeeds."""
  expr = cst.parse_expression("in_top_k(preds, targets, 5)")
  assert isinstance(expr, cst.Call)

  semantics = mock.MagicMock()
  semantics.resolve_variant.return_value = {"api": "torch.topk"}
  cfg = RuntimeConfig(source_framework="tensorflow", target_framework="torch")
  ctx = HookContext(semantics=semantics, config=cfg)

  res = in_top_k_plugin(expr, ctx)
  assert isinstance(res, cst.BaseExpression)
  code = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=res)])]).code.strip()
  assert code == "(targets.unsqueeze(-1) == torch.topk(preds, 5).indices).any(dim=-1)"


def test_in_top_k_plugin_fallback_and_non_torch_abort() -> None:
  """Verify `in_top_k_plugin` falls back to `target_fw.topk` or aborts on NumPy-compatible array targets."""
  expr = cst.parse_expression("in_top_k(preds, targets, k)")
  assert isinstance(expr, cst.Call)

  # Fallback when lookup_api returns None and target does not use numpy arrays
  semantics = mock.MagicMock()
  semantics.resolve_variant.return_value = None
  semantics.get_framework_config.return_value = {"plugin_traits": PluginTraits(has_numpy_compatible_arrays=False)}
  cfg = RuntimeConfig(source_framework="tensorflow", target_framework="torch")
  ctx = HookContext(semantics=semantics, config=cfg)

  res = in_top_k_plugin(expr, ctx)
  assert isinstance(res, cst.BaseExpression)
  code = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=res)])]).code.strip()
  assert code == "(targets.unsqueeze(-1) == torch.topk(preds, k).indices).any(dim=-1)"

  # Abort when target has numpy-compatible arrays and no TopK mapping
  semantics_jax = mock.MagicMock()
  semantics_jax.resolve_variant.return_value = None
  semantics_jax.get_framework_config.return_value = {"plugin_traits": PluginTraits(has_numpy_compatible_arrays=True)}
  cfg_jax = RuntimeConfig(source_framework="tensorflow", target_framework="jax")
  ctx_jax = HookContext(semantics=semantics_jax, config=cfg_jax)
  assert in_top_k_plugin(expr, ctx_jax) is expr
