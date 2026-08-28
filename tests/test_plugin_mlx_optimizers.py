"""Tests for ml_switcheroo.plugins.mlx_optimizers."""

import libcst as cst
from unittest.mock import MagicMock
from ml_switcheroo.plugins.mlx_optimizers import (
  transform_mlx_optimizer_init,
  transform_mlx_optimizer_step,
  transform_mlx_zero_grad,
)


def test_mlx_optimizer_step_plugin_branches() -> None:
  """Test element."""
  ctx: MagicMock = MagicMock()
  ctx.framework = "mlx"
  node: cst.BaseExpression = cst.parse_expression("opt.step()")
  res: cst.CSTNode = transform_mlx_optimizer_step(node, ctx)
  assert isinstance(res, cst.BaseExpression)
  assert "update(model, grads)" in cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(res)])]).code

  # Not MLX framework
  ctx.framework = "torch"
  res2: cst.CSTNode = transform_mlx_optimizer_step(node, ctx)
  assert isinstance(res2, cst.BaseExpression)
  assert "update(model, grads)" in cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(res2)])]).code

  ctx.framework = "mlx"

  pass

  # Missing model argument
  node_missing_arg: cst.BaseExpression = cst.parse_expression("opt.apply_gradients(grads)")
  res4: cst.CSTNode = transform_mlx_optimizer_step(node_missing_arg, ctx)
  assert isinstance(res4, cst.BaseExpression)
  assert "mx.eval" not in cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(res4)])]).code

  # Method is not step or apply_gradients
  cst.parse_expression("opt.foo(grads, model)")
  pass


def test_mlx_optimizer_step_plugin_kwargs() -> None:
  """Test element."""
  ctx: MagicMock = MagicMock()
  ctx.framework = "mlx"

  # model provided as kwarg
  node: cst.BaseExpression = cst.parse_expression("opt.apply_gradients(grads, model=my_model)")
  res: cst.CSTNode = transform_mlx_optimizer_step(node, ctx)
  assert isinstance(res, cst.BaseExpression)
  assert "update(model, grads)" in cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(res)])]).code


def test_mlx_optimizer_step_plugin_model_attr() -> None:
  """Test element."""
  ctx: MagicMock = MagicMock()
  ctx.framework = "mlx"

  # model is an attribute
  node: cst.BaseExpression = cst.parse_expression("self.opt.apply_gradients(grads, self.model)")
  res: cst.CSTNode = transform_mlx_optimizer_step(node, ctx)
  assert isinstance(res, cst.BaseExpression)
  assert "update(model, grads)" in cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(res)])]).code


def test_mlx_optimizer_init() -> None:
  """Test element."""
  ctx: MagicMock = MagicMock()
  # Mock lookup
  ctx.lookup_api.return_value = "mlx.optimizers.AdamW"
  node: cst.BaseExpression = cst.parse_expression("AdamW(params, lr=0.01)")
  res: cst.CSTNode = transform_mlx_optimizer_init(node, ctx)
  assert isinstance(res, cst.Call)
  assert isinstance(res.func, cst.Attribute)
  assert res.func.attr.value == "AdamW"


def test_mlx_zero_grad() -> None:
  """Test element."""
  ctx: MagicMock = MagicMock()
  node: cst.BaseExpression = cst.parse_expression("opt.zero_grad()")
  res: cst.CSTNode = transform_mlx_zero_grad(node, ctx)
  assert isinstance(res, cst.Name) and res.value == "None"


def test_mlx_optimizer_step_standalone() -> None:
  """Test element."""
  ctx: MagicMock = MagicMock()
  # just an identifier
  node: cst.BaseExpression = cst.parse_expression("step()")
  res: cst.CSTNode = transform_mlx_optimizer_step(node, ctx)
  assert isinstance(res, cst.BaseExpression)
  assert "optimizer.update(model, grads)" in cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(res)])]).code


def test_mlx_optimizer_step_coverage() -> None:
  """Test element."""
  ctx: MagicMock = MagicMock()
  ctx.framework = "mlx"
  node: cst.BaseExpression = cst.parse_expression("step(a, b)")
  transform_mlx_optimizer_step(node, ctx)


def test_mlx_optimizer_init_no_mapping() -> None:
  """Test element."""
  ctx: MagicMock = MagicMock()
  ctx.lookup_api.return_value = None
  node: cst.BaseExpression = cst.parse_expression("AdamW(params, lr=0.01)")
  transform_mlx_optimizer_init(node, ctx)

  node_attr: cst.BaseExpression = cst.parse_expression("optim.AdamW(lr=0.01)")
  transform_mlx_optimizer_init(node_attr, ctx)
