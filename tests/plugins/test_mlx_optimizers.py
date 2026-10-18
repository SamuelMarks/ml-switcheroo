"""Test suite for mlx optimizers."""

import typing

from typing import cast
from unittest.mock import MagicMock

import libcst as cst

from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.mlx_optimizers import (
  _create_dotted_name,
  transform_mlx_optimizer_init,
  transform_mlx_optimizer_step,
  transform_mlx_zero_grad,
)


def test_create_dotted_name() -> None:
  """Verify create dotted name behavior."""
  node: typing.Any = _create_dotted_name("mlx.optimizers.Adam")
  assert isinstance(node, cst.Attribute)
  assert node.attr.value == "Adam"
  assert isinstance(node.value, cst.Attribute)
  assert node.value.attr.value == "optimizers"
  assert isinstance(node.value.value, cst.Name)
  assert node.value.value.value == "mlx"


def test_transform_optimizer_init_with_lookup() -> None:
  """Verify transform optimizer init with lookup behavior."""
  ctx: typing.Any = MagicMock(spec=HookContext)
  ctx.current_op_id = "Adam"
  ctx.lookup_api.return_value = "mlx.optimizers.AdamW"

  code: typing.Any = "optim.Adam(model.parameters(), lr=0.01, beta1=0.9)"
  module: typing.Any = cst.parse_module(code)
  call_node: typing.Any = cast(cst.Call, module.body[0].body[0].value)

  transformed: typing.Any = transform_mlx_optimizer_init(call_node, ctx)

  assert isinstance(transformed.func, cst.Attribute)
  assert transformed.func.attr.value == "AdamW"
  assert len(transformed.args) == 2
  assert transformed.args[1].keyword is not None
  assert transformed.args[1].keyword.value == "beta1"


def test_transform_optimizer_init_fallback_attribute() -> None:
  """Verify transform optimizer init fallback attribute behavior."""
  ctx: typing.Any = MagicMock(spec=HookContext)
  ctx.current_op_id = None
  ctx.lookup_api.return_value = None

  code: typing.Any = "optim.Adam(lr=0.01)"
  module: typing.Any = cst.parse_module(code)
  call_node: typing.Any = cast(cst.Call, module.body[0].body[0].value)

  res: typing.Any = transform_mlx_optimizer_init(call_node, ctx)
  assert isinstance(res, cst.Call)
  assert "mlx.optimizers.Adam" in cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=res)])]).code


def test_transform_optimizer_init_fallback_name() -> None:
  """Verify transform optimizer init fallback name behavior."""
  ctx: typing.Any = MagicMock(spec=HookContext)
  ctx.current_op_id = None
  ctx.lookup_api.return_value = None

  code: typing.Any = "Adam(lr=0.01)"
  module: typing.Any = cst.parse_module(code)
  call_node: typing.Any = cast(cst.Call, module.body[0].body[0].value)

  res: typing.Any = transform_mlx_optimizer_init(call_node, ctx)
  assert isinstance(res, cst.Call)
  assert "mlx.optimizers.Adam" in cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=res)])]).code


def test_transform_optimizer_init_fallback_other() -> None:
  """Verify transform optimizer init fallback other behavior."""
  ctx: typing.Any = MagicMock(spec=HookContext)
  ctx.current_op_id = None
  ctx.lookup_api.return_value = None

  code: typing.Any = "(get_opt())(lr=0.01)"
  module: typing.Any = cst.parse_module(code)
  stmt: typing.Any = cast(cst.SimpleStatementLine, module.body[0])
  expr: typing.Any = cast(cst.Expr, stmt.body[0])
  call_node: typing.Any = cast(cst.Call, expr.value)

  res: typing.Any = transform_mlx_optimizer_init(call_node, ctx)
  assert isinstance(res, cst.Call)
  assert "mlx.optimizers.Adam" in cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=res)])]).code


def test_transform_optimizer_step_attribute() -> None:
  """Verify transform optimizer step attribute behavior."""
  ctx: typing.Any = MagicMock(spec=HookContext)

  code: typing.Any = "my_opt.step()"
  module: typing.Any = cst.parse_module(code)
  call_node: typing.Any = cast(cst.Call, module.body[0].body[0].value)

  transformed: typing.Any = transform_mlx_optimizer_step(call_node, ctx)

  assert isinstance(transformed, cst.Call)

  assert isinstance(transformed.func, cst.Attribute)
  assert isinstance(transformed.func.value, cst.Name)
  assert transformed.func.value.value == "my_opt"
  assert transformed.func.attr.value == "update"
  assert len(transformed.args) == 2


def test_transform_optimizer_step_name() -> None:
  """Verify transform optimizer step name behavior."""
  ctx: typing.Any = MagicMock(spec=HookContext)

  code: typing.Any = "step()"
  module: typing.Any = cst.parse_module(code)
  call_node: typing.Any = cast(cst.Call, module.body[0].body[0].value)

  transformed: typing.Any = transform_mlx_optimizer_step(call_node, ctx)

  assert isinstance(transformed, cst.Call)

  assert isinstance(transformed.func, cst.Attribute)
  assert isinstance(transformed.func.value, cst.Name)
  assert transformed.func.value.value == "optimizer"
  assert transformed.func.attr.value == "update"
  assert len(transformed.args) == 2


def test_transform_zero_grad() -> None:
  """Verify transform zero grad behavior."""
  ctx: typing.Any = MagicMock(spec=HookContext)

  code: typing.Any = "opt.zero_grad()"
  module: typing.Any = cst.parse_module(code)
  call_node: typing.Any = cast(cst.Call, module.body[0].body[0].value)

  transformed: typing.Any = transform_mlx_zero_grad(call_node, ctx)

  assert isinstance(transformed, cst.Name)
  assert transformed.value == "None"


def test_transform_optimizer_init_positional_args() -> None:
  """Verify transform optimizer init positional args behavior."""
  ctx: typing.Any = MagicMock(spec=HookContext)
  ctx.current_op_id = "Adam"
  ctx.lookup_api.return_value = "mlx.optimizers.AdamW"

  code: typing.Any = "optim.Adam(model.parameters(), 0.01, 0.9)"
  module: typing.Any = cst.parse_module(code)
  call_node: typing.Any = cast(cst.Call, module.body[0].body[0].value)

  transformed: typing.Any = transform_mlx_optimizer_init(call_node, ctx)

  assert isinstance(transformed.func, cst.Attribute)
  assert transformed.func.attr.value == "AdamW"
  assert len(transformed.args) == 2


def test_transform_optimizer_init_kwarg_no_value() -> None:
  """Verify transform optimizer init kwarg no value behavior."""
  ctx: typing.Any = MagicMock(spec=HookContext)
  ctx.current_op_id = None
  ctx.lookup_api.return_value = None

  code: typing.Any = "optim.Adam(**kwargs)"
  module: typing.Any = cst.parse_module(code)
  call_node: typing.Any = cast(cst.Call, module.body[0].body[0].value)

  res: typing.Any = transform_mlx_optimizer_init(call_node, ctx)
  assert isinstance(res, cst.Call)
  assert len(res.args) == 0


def test_transform_optimizer_init_lr_rename_no_keyword() -> None:
  """Verify transform optimizer init lr rename no keyword behavior."""
  ctx: typing.Any = MagicMock()
  ctx.current_op_id = "Adam"
  ctx.lookup_api.return_value = "mlx.optimizers.Adam"
  code: typing.Any = "optim.Adam(model.parameters(), lr=0.01)"
  module: typing.Any = cst.parse_module(code)
  call_node: typing.Any = cast(cst.Call, module.body[0].body[0].value)

  transformed: typing.Any = transform_mlx_optimizer_init(call_node, ctx)
  assert transformed.args[0].keyword is not None
  assert transformed.args[0].keyword.value == "learning_rate"


def test_transform_optimizer_init_no_args() -> None:
  """Verify transform optimizer init no args behavior."""
  ctx: typing.Any = MagicMock()
  ctx.current_op_id = "Adam"
  ctx.lookup_api.return_value = "mlx.optimizers.Adam"
  code: typing.Any = "optim.Adam()"
  module: typing.Any = cst.parse_module(code)
  call_node: typing.Any = cast(cst.Call, module.body[0].body[0].value)

  transformed: typing.Any = transform_mlx_optimizer_init(call_node, ctx)
  assert len(transformed.args) == 0


def test_transform_optimizer_init_lr_rename_no_keyword_true() -> None:
  """Verify transform optimizer init lr rename no keyword true behavior."""
  ctx: typing.Any = MagicMock()
  ctx.current_op_id = "Adam"
  ctx.lookup_api.return_value = "mlx.optimizers.Adam"
  code: typing.Any = "optim.Adam(lr=0.01)"  # no positional args
  module: typing.Any = cst.parse_module(code)
  call_node: typing.Any = cast(cst.Call, module.body[0].body[0].value)

  transformed: typing.Any = transform_mlx_optimizer_init(call_node, ctx)
  assert transformed.args[0].keyword is not None
  assert transformed.args[0].keyword.value == "learning_rate"


def test_transform_optimizer_init_lr_rename_no_keyword_false() -> None:
  """Verify transform optimizer init lr rename no keyword false behavior."""
  ctx: typing.Any = MagicMock()
  ctx.current_op_id = "Adam"
  ctx.lookup_api.return_value = "mlx.optimizers.Adam"
  code: typing.Any = "optim.Adam(model.parameters(), 0.01)"  # no positional args
  module: typing.Any = cst.parse_module(code)
  call_node: typing.Any = cast(cst.Call, module.body[0].body[0].value)

  transformed: typing.Any = transform_mlx_optimizer_init(call_node, ctx)
  assert isinstance(transformed.args[0].value, cst.Float)
  assert transformed.args[0].value.value == "0.01"


def test_transform_optimizer_init_lr_rename_no_keyword_true_already() -> None:
  """Verify transform optimizer init lr rename no keyword true already behavior."""
  ctx: typing.Any = MagicMock()
  ctx.current_op_id = "Adam"
  ctx.lookup_api.return_value = "mlx.optimizers.Adam"
  code: typing.Any = "optim.Adam(learning_rate=0.01)"
  module: typing.Any = cst.parse_module(code)
  call_node: typing.Any = cast(cst.Call, module.body[0].body[0].value)

  transformed: typing.Any = transform_mlx_optimizer_init(call_node, ctx)
  assert transformed.args[0].keyword is not None
  assert transformed.args[0].keyword.value == "learning_rate"


def test_transform_optimizer_init_lr_rename_no_keyword_true_actually() -> None:
  """Verify transform optimizer init lr rename no keyword true actually behavior."""
  ctx: typing.Any = MagicMock()
  ctx.current_op_id = "Adam"
  ctx.lookup_api.return_value = "mlx.optimizers.Adam"
  code: typing.Any = "optim.Adam(lr=0.01)"
  module: typing.Any = cst.parse_module(code)
  call_node: typing.Any = cast(cst.Call, module.body[0].body[0].value)
  transformed: typing.Any = transform_mlx_optimizer_init(call_node, ctx)
  assert transformed.args[0].keyword is not None
  assert transformed.args[0].keyword.value == "learning_rate"
