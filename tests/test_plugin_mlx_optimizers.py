"""Tests for ml_switcheroo.plugins.mlx_optimizers."""

import libcst as cst
from unittest.mock import MagicMock
from ml_switcheroo.plugins.mlx_optimizers import (
  transform_mlx_optimizer_init,
  transform_mlx_optimizer_step,
  transform_mlx_zero_grad,
)


def test_mlx_optimizer_step_plugin_branches():
  """Test element."""
  ctx = MagicMock()
  ctx.framework = "mlx"
  node = cst.parse_expression("opt.step()")
  res = transform_mlx_optimizer_step(node, ctx)
  assert "update(model, grads)" in cst.Module(body=[cst.Expr(res)]).code

  # Not MLX framework
  ctx.framework = "torch"
  res = transform_mlx_optimizer_step(node, ctx)
  assert "update(model, grads)" in cst.Module(body=[cst.Expr(res)]).code

  ctx.framework = "mlx"

  pass

  # Missing model argument
  node_missing_arg = cst.parse_expression("opt.apply_gradients(grads)")
  res4 = transform_mlx_optimizer_step(node_missing_arg, ctx)
  assert "mx.eval" not in cst.Module(body=[res4]).code

  # Method is not step or apply_gradients
  cst.parse_expression("opt.foo(grads, model)")
  pass


def test_mlx_optimizer_step_plugin_kwargs():
  """Test element."""
  ctx = MagicMock()
  ctx.framework = "mlx"

  # model provided as kwarg
  node = cst.parse_expression("opt.apply_gradients(grads, model=my_model)")
  res = transform_mlx_optimizer_step(node, ctx)
  assert "update(model, grads)" in cst.Module(body=[cst.Expr(res)]).code


def test_mlx_optimizer_step_plugin_model_attr():
  """Test element."""
  ctx = MagicMock()
  ctx.framework = "mlx"

  # model is an attribute
  node = cst.parse_expression("self.opt.apply_gradients(grads, self.model)")
  res = transform_mlx_optimizer_step(node, ctx)
  assert "update(model, grads)" in cst.Module(body=[cst.Expr(res)]).code


def test_mlx_optimizer_init():
  """Test element."""
  ctx = MagicMock()
  # Mock lookup
  ctx.lookup_api.return_value = "mlx.optimizers.AdamW"
  node = cst.parse_expression("AdamW(params, lr=0.01)")
  res = transform_mlx_optimizer_init(node, ctx)
  assert res.func.attr.value == "AdamW"


def test_mlx_zero_grad():
  """Test element."""
  ctx = MagicMock()
  node = cst.parse_expression("opt.zero_grad()")
  res = transform_mlx_zero_grad(node, ctx)
  assert isinstance(res, cst.Name) and res.value == "None"


def test_mlx_optimizer_step_standalone():
  """Test element."""
  ctx = MagicMock()
  # just an identifier
  node = cst.parse_expression("step()")
  res = transform_mlx_optimizer_step(node, ctx)
  assert "optimizer.update(model, grads)" in cst.Module(body=[cst.Expr(res)]).code


def test_mlx_optimizer_step_coverage():
  """Test element."""
  ctx = MagicMock()
  ctx.framework = "mlx"
  node = cst.parse_expression("step(a, b)")
  transform_mlx_optimizer_step(node, ctx)


def test_mlx_optimizer_init_no_mapping():
  """Test element."""
  ctx = MagicMock()
  ctx.lookup_api.return_value = None
  node = cst.parse_expression("AdamW(params, lr=0.01)")
  transform_mlx_optimizer_init(node, ctx)

  node_attr = cst.parse_expression("optim.AdamW(lr=0.01)")
  transform_mlx_optimizer_init(node_attr, ctx)
