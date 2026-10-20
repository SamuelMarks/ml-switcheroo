"""Tests for ml_switcheroo.plugins.mlx_optimizers."""

from unittest.mock import MagicMock

import libcst as cst

from ml_switcheroo.plugins.mlx_optimizers import (
  transform_mlx_optimizer_init,
  transform_mlx_optimizer_step,
  transform_mlx_zero_grad,
)


def test_mlx_optimizer_step_plugin_branches() -> None:
  """Verify mlx optimizer step plugin branches behavior."""
  ctx: MagicMock = MagicMock()
  ctx.framework = "mlx"
  node: cst.Call = cst.ensure_type(cst.parse_expression("opt.step()"), cst.Call)
  res: cst.CSTNode = transform_mlx_optimizer_step(node, ctx)  # type: ignore[assignment]
  assert isinstance(res, cst.Call)
  assert "update(model, grads)" in cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(res)])]).code

  # Not MLX framework
  ctx.framework = "torch"
  res2: cst.CSTNode = transform_mlx_optimizer_step(node, ctx)  # type: ignore[assignment]
  assert isinstance(res2, cst.Call)
  assert "update(model, grads)" in cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(res2)])]).code

  ctx.framework = "mlx"

  pass

  # Missing model argument
  node_missing_arg: cst.Call = cst.ensure_type(cst.parse_expression("opt.apply_gradients(grads)"), cst.Call)
  res4: cst.CSTNode = transform_mlx_optimizer_step(node_missing_arg, ctx)  # type: ignore[assignment]
  assert isinstance(res4, cst.Call)
  assert "mx.eval" not in cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(res4)])]).code

  # Method is not step or apply_gradients
  cst.parse_expression("opt.foo(grads, model)")
  pass


def test_mlx_optimizer_step_plugin_kwargs() -> None:
  """Verify mlx optimizer step plugin kwargs behavior."""
  ctx: MagicMock = MagicMock()
  ctx.framework = "mlx"

  # model provided as kwarg
  node: cst.Call = cst.ensure_type(cst.parse_expression("opt.apply_gradients(grads, model=my_model)"), cst.Call)
  res: cst.CSTNode = transform_mlx_optimizer_step(node, ctx)  # type: ignore[assignment]
  assert isinstance(res, cst.Call)
  assert "update(model, grads)" in cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(res)])]).code


def test_mlx_optimizer_step_plugin_model_attr() -> None:
  """Verify mlx optimizer step plugin model attr behavior."""
  ctx: MagicMock = MagicMock()
  ctx.framework = "mlx"

  # model is an attribute
  node: cst.Call = cst.ensure_type(cst.parse_expression("self.opt.apply_gradients(grads, self.model)"), cst.Call)
  res: cst.CSTNode = transform_mlx_optimizer_step(node, ctx)  # type: ignore[assignment]
  assert isinstance(res, cst.Call)
  assert "update(model, grads)" in cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(res)])]).code


def test_mlx_optimizer_init() -> None:
  """Verify mlx optimizer init behavior."""
  ctx: MagicMock = MagicMock()
  # Mock lookup
  ctx.lookup_api.return_value = "mlx.optimizers.AdamW"
  node: cst.Call = cst.ensure_type(cst.parse_expression("AdamW(params, lr=0.01)"), cst.Call)
  res: cst.CSTNode = transform_mlx_optimizer_init(node, ctx)
  assert isinstance(res, cst.Call)
  assert isinstance(res.func, cst.Attribute)
  assert res.func.attr.value == "AdamW"


def test_mlx_zero_grad() -> None:
  """Verify mlx zero grad behavior."""
  ctx: MagicMock = MagicMock()
  node: cst.Call = cst.ensure_type(cst.parse_expression("opt.zero_grad()"), cst.Call)
  res: cst.CSTNode = transform_mlx_zero_grad(node, ctx)  # type: ignore[assignment]
  assert isinstance(res, cst.Name) and res.value == "None"


def test_mlx_optimizer_step_standalone() -> None:
  """Verify mlx optimizer step standalone behavior."""
  ctx: MagicMock = MagicMock()
  # just an identifier
  node: cst.Call = cst.ensure_type(cst.parse_expression("step()"), cst.Call)
  res: cst.CSTNode = transform_mlx_optimizer_step(node, ctx)  # type: ignore[assignment]
  assert isinstance(res, cst.Call)
  assert "optimizer.update(model, grads)" in cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(res)])]).code


def test_mlx_optimizer_step_coverage() -> None:
  """Verify mlx optimizer step coverage behavior."""
  ctx: MagicMock = MagicMock()
  ctx.framework = "mlx"
  node: cst.Call = cst.ensure_type(cst.parse_expression("step(a, b)"), cst.Call)
  _res = transform_mlx_optimizer_step(node, ctx)
  assert isinstance(_res, cst.Call)


def test_mlx_optimizer_init_no_mapping() -> None:
  """Verify mlx optimizer init no mapping behavior."""
  ctx: MagicMock = MagicMock()
  ctx.lookup_api.return_value = None
  node: cst.Call = cst.ensure_type(cst.parse_expression("AdamW(params, lr=0.01)"), cst.Call)
  transform_mlx_optimizer_init(node, ctx)

  node_attr: cst.Call = cst.ensure_type(cst.parse_expression("optim.AdamW(lr=0.01)"), cst.Call)
  _res = transform_mlx_optimizer_init(node_attr, ctx)
  assert isinstance(_res, cst.Call)
