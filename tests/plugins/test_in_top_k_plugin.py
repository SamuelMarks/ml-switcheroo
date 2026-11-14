"""Unit tests for `in_top_k_plugin` (`ml_switcheroo.plugins.in_top_k_plugin`)."""

from unittest import mock

import libcst as cst

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.dsl import FrameworkVariant
from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.in_top_k_plugin import in_top_k_plugin
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.semantics.schema import PluginTraits


def _expr_to_code(node: cst.CSTNode) -> str:
  """Render a rewritten CST expression back to stripped Python code.

  Args:
      node: Rewritten LibCST expression node.

  Returns:
      Formatted Python expression string.
  """
  assert isinstance(node, cst.BaseExpression)
  return cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=node)])]).code.strip()


def test_in_top_k_plugin_passthrough_cases() -> None:
  """Verify `in_top_k_plugin` returns the input call unchanged on <3 args or incompatible targets."""
  mgr = SemanticsManager()
  ctx_jax = HookContext(config=RuntimeConfig(source_framework="jax", target_framework="jax"), semantics=mgr)

  two_arg_call = cst.Call(func=cst.Name("in_top_k"), args=[cst.Arg(cst.Name("p")), cst.Arg(cst.Name("t"))])
  assert in_top_k_plugin(two_arg_call, ctx_jax) is two_arg_call

  ctx_placeholder = HookContext(
    config=RuntimeConfig(source_framework="jax", target_framework="target_placeholder"),
    semantics=mgr,
  )
  three_arg_call = cst.ensure_type(cst.parse_expression("in_top_k(p, t, 5)"), cst.Call)
  assert in_top_k_plugin(three_arg_call, ctx_placeholder) is three_arg_call


def test_in_top_k_plugin_torch_and_lookup_fallbacks() -> None:
  """Verify `in_top_k_plugin` rewrites `in_top_k(preds, targets, k)` using `topk(...).indices`."""
  mgr = SemanticsManager()
  ctx_torch = HookContext(config=RuntimeConfig(source_framework="jax", target_framework="torch"), semantics=mgr)
  expr = cst.ensure_type(cst.parse_expression("in_top_k(preds, targets, 5)"), cst.Call)

  rewritten = in_top_k_plugin(expr, ctx_torch)
  assert _expr_to_code(rewritten) == "(targets.unsqueeze(-1) == torch.topk(preds, 5).indices).any(dim=-1)"

  # Fallback when `resolve_variant` returns None and target does not use NumPy-compatible arrays
  semantics_fallback = mock.MagicMock(spec=SemanticsManager)
  semantics_fallback.resolve_variant.return_value = None
  semantics_fallback.get_framework_config.return_value = {
    "plugin_traits": PluginTraits(has_numpy_compatible_arrays=False)
  }
  ctx_fallback = HookContext(
    semantics=semantics_fallback,
    config=RuntimeConfig(source_framework="tensorflow", target_framework="torch"),
  )
  assert _expr_to_code(in_top_k_plugin(expr, ctx_fallback)) == (
    "(targets.unsqueeze(-1) == torch.topk(preds, 5).indices).any(dim=-1)"
  )

  # Explicit `FrameworkVariant` lookup result
  semantics_explicit = mock.MagicMock(spec=SemanticsManager)
  semantics_explicit.resolve_variant.return_value = FrameworkVariant(api="torch.topk")
  ctx_explicit = HookContext(
    semantics=semantics_explicit,
    config=RuntimeConfig(source_framework="tensorflow", target_framework="torch"),
  )
  assert _expr_to_code(in_top_k_plugin(expr, ctx_explicit)) == (
    "(targets.unsqueeze(-1) == torch.topk(preds, 5).indices).any(dim=-1)"
  )

  # Abort when target has NumPy-compatible arrays and no `TopK` variant is mapped
  semantics_np = mock.MagicMock(spec=SemanticsManager)
  semantics_np.resolve_variant.return_value = None
  semantics_np.get_framework_config.return_value = {"plugin_traits": PluginTraits(has_numpy_compatible_arrays=True)}
  ctx_np = HookContext(
    semantics=semantics_np,
    config=RuntimeConfig(source_framework="tensorflow", target_framework="jax"),
  )
  assert in_top_k_plugin(expr, ctx_np) is expr
