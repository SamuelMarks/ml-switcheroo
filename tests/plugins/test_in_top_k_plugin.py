"""Tests for the in_top_k plugin."""

import libcst as cst
from ml_switcheroo.plugins.in_top_k_plugin import in_top_k_plugin
from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager


def test_in_top_k_plugin_passthrough() -> None:
  """Verifies the plugin passes through when framework is not torch or args < 3."""
  mgr = SemanticsManager()
  cfg = RuntimeConfig(source_framework="jax", target_framework="jax")
  ctx = HookContext(config=cfg, semantics=mgr)

  node = cst.Call(func=cst.Name("in_top_k"), args=[cst.Arg(cst.Name("p")), cst.Arg(cst.Name("t"))])
  res = in_top_k_plugin(node, ctx)
  assert res is node

  cfg = RuntimeConfig(source_framework="jax", target_framework="target_placeholder")
  ctx = HookContext(config=cfg, semantics=mgr)
  node = cst.Call(
    func=cst.Name("in_top_k"), args=[cst.Arg(cst.Name("p")), cst.Arg(cst.Name("t")), cst.Arg(cst.Integer("5"))]
  )
  res = in_top_k_plugin(node, ctx)
  assert res is node


def test_in_top_k_plugin_torch() -> None:
  """Verifies the plugin transforms correctly for torch."""
  mgr = SemanticsManager()
  cfg = RuntimeConfig(source_framework="jax", target_framework="torch")

  ctx = HookContext(config=cfg, semantics=mgr)

  node = cst.parse_expression("in_top_k(preds, targets, 5)")
  res = in_top_k_plugin(node, ctx)

  # verify code output
  module = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=res)])])
  assert "targets.unsqueeze(-1)" in module.code
