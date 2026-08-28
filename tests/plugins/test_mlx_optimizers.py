"""Docstring."""

import libcst as cst
from typing import Union, cast
from unittest.mock import MagicMock

from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.mlx_optimizers import (
  _create_dotted_name,
  transform_mlx_optimizer_init,
  transform_mlx_optimizer_step,
  transform_mlx_zero_grad,
)


def test_create_dotted_name() -> None:
  """Docstring."""
  node: Union[cst.Name, cst.Attribute] = _create_dotted_name("mlx.optimizers.Adam")
  assert isinstance(node, cst.Attribute)
  assert node.attr.value == "Adam"
  assert isinstance(node.value, cst.Attribute)
  assert node.value.attr.value == "optimizers"
  assert isinstance(node.value.value, cst.Name)
  assert node.value.value.value == "mlx"


def test_transform_optimizer_init_with_lookup() -> None:
  """Docstring."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.current_op_id = "Adam"
  ctx.lookup_api.return_value = "mlx.optimizers.AdamW"

  code: str = "optim.Adam(model.parameters(), lr=0.01, beta1=0.9)"
  module: cst.Module = cst.parse_module(code)
  call_node: cst.Call = cast(cst.Call, module.body[0].body[0].value)

  transformed: cst.Call = transform_mlx_optimizer_init(call_node, ctx)

  assert isinstance(transformed.func, cst.Attribute)
  assert transformed.func.attr.value == "AdamW"
  assert len(transformed.args) == 2
  assert transformed.args[1].keyword is not None
  assert transformed.args[1].keyword.value == "beta1"


def test_transform_optimizer_init_fallback_attribute() -> None:
  """Docstring."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.current_op_id = None
  ctx.lookup_api.return_value = None

  code: str = "optim.Adam(lr=0.01)"
  module: cst.Module = cst.parse_module(code)
  call_node: cst.Call = cast(cst.Call, module.body[0].body[0].value)

  transform_mlx_optimizer_init(call_node, ctx)


def test_transform_optimizer_init_fallback_name() -> None:
  """Docstring."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.current_op_id = None
  ctx.lookup_api.return_value = None

  code: str = "Adam(lr=0.01)"
  module: cst.Module = cst.parse_module(code)
  call_node: cst.Call = cast(cst.Call, module.body[0].body[0].value)

  transform_mlx_optimizer_init(call_node, ctx)


def test_transform_optimizer_step_attribute() -> None:
  """Docstring."""
  ctx: MagicMock = MagicMock(spec=HookContext)

  code: str = "my_opt.step()"
  module: cst.Module = cst.parse_module(code)
  call_node: cst.Call = cast(cst.Call, module.body[0].body[0].value)

  transformed: cst.Call = transform_mlx_optimizer_step(call_node, ctx)

  assert isinstance(transformed, cst.Call)

  assert isinstance(transformed.func, cst.Attribute)
  assert isinstance(transformed.func.value, cst.Name)
  assert transformed.func.value.value == "my_opt"
  assert transformed.func.attr.value == "update"
  assert len(transformed.args) == 2


def test_transform_optimizer_step_name() -> None:
  """Docstring."""
  ctx: MagicMock = MagicMock(spec=HookContext)

  code: str = "step()"
  module: cst.Module = cst.parse_module(code)
  call_node: cst.Call = cast(cst.Call, module.body[0].body[0].value)

  transformed: cst.Call = transform_mlx_optimizer_step(call_node, ctx)

  assert isinstance(transformed, cst.Call)

  assert isinstance(transformed.func, cst.Attribute)
  assert isinstance(transformed.func.value, cst.Name)
  assert transformed.func.value.value == "optimizer"
  assert transformed.func.attr.value == "update"
  assert len(transformed.args) == 2


def test_transform_zero_grad() -> None:
  """Docstring."""
  ctx: MagicMock = MagicMock(spec=HookContext)

  code: str = "opt.zero_grad()"
  module: cst.Module = cst.parse_module(code)
  call_node: cst.Call = cast(cst.Call, module.body[0].body[0].value)

  transformed: cst.Name = transform_mlx_zero_grad(call_node, ctx)

  assert isinstance(transformed, cst.Name)
  assert transformed.value == "None"


def test_transform_optimizer_init_positional_args() -> None:
  """Docstring."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.current_op_id = "Adam"
  ctx.lookup_api.return_value = "mlx.optimizers.AdamW"

  code: str = "optim.Adam(model.parameters(), 0.01, 0.9)"
  module: cst.Module = cst.parse_module(code)
  call_node: cst.Call = cast(cst.Call, module.body[0].body[0].value)

  transformed: cst.Call = transform_mlx_optimizer_init(call_node, ctx)

  assert isinstance(transformed.func, cst.Attribute)
  assert transformed.func.attr.value == "AdamW"
  assert len(transformed.args) == 2


def test_transform_optimizer_init_kwarg_no_value() -> None:
  """Docstring."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.current_op_id = None
  ctx.lookup_api.return_value = None

  code: str = "optim.Adam(**kwargs)"
  module: cst.Module = cst.parse_module(code)
  call_node: cst.Call = cast(cst.Call, module.body[0].body[0].value)

  transform_mlx_optimizer_init(call_node, ctx)


def test_transform_optimizer_init_lr_rename_no_keyword() -> None:
  """Docstring."""
  ctx: MagicMock = MagicMock()
  ctx.current_op_id = "Adam"
  ctx.lookup_api.return_value = "mlx.optimizers.Adam"
  code: str = "optim.Adam(model.parameters(), lr=0.01)"
  module: cst.Module = cst.parse_module(code)
  call_node: cst.Call = cast(cst.Call, module.body[0].body[0].value)

  transformed: cst.Call = transform_mlx_optimizer_init(call_node, ctx)
  assert transformed.args[0].keyword is not None
  assert transformed.args[0].keyword.value == "learning_rate"


def test_transform_optimizer_init_no_args() -> None:
  """Docstring."""
  ctx: MagicMock = MagicMock()
  ctx.current_op_id = "Adam"
  ctx.lookup_api.return_value = "mlx.optimizers.Adam"
  code: str = "optim.Adam()"
  module: cst.Module = cst.parse_module(code)
  call_node: cst.Call = cast(cst.Call, module.body[0].body[0].value)

  transformed: cst.Call = transform_mlx_optimizer_init(call_node, ctx)
  assert len(transformed.args) == 0


def test_transform_optimizer_init_lr_rename_no_keyword_true() -> None:
  """Docstring."""
  ctx: MagicMock = MagicMock()
  ctx.current_op_id = "Adam"
  ctx.lookup_api.return_value = "mlx.optimizers.Adam"
  code: str = "optim.Adam(lr=0.01)"  # no positional args
  module: cst.Module = cst.parse_module(code)
  call_node: cst.Call = cast(cst.Call, module.body[0].body[0].value)

  transformed: cst.Call = transform_mlx_optimizer_init(call_node, ctx)
  assert transformed.args[0].keyword is not None
  assert transformed.args[0].keyword.value == "learning_rate"


def test_transform_optimizer_init_lr_rename_no_keyword_false() -> None:
  """Docstring."""
  ctx: MagicMock = MagicMock()
  ctx.current_op_id = "Adam"
  ctx.lookup_api.return_value = "mlx.optimizers.Adam"
  code: str = "optim.Adam(model.parameters(), 0.01)"  # no positional args
  module: cst.Module = cst.parse_module(code)
  call_node: cst.Call = cast(cst.Call, module.body[0].body[0].value)

  transformed: cst.Call = transform_mlx_optimizer_init(call_node, ctx)
  assert isinstance(transformed.args[0].value, cst.Float)
  assert transformed.args[0].value.value == "0.01"


def test_transform_optimizer_init_lr_rename_no_keyword_true_already() -> None:
  """Docstring."""
  ctx: MagicMock = MagicMock()
  ctx.current_op_id = "Adam"
  ctx.lookup_api.return_value = "mlx.optimizers.Adam"
  code: str = "optim.Adam(learning_rate=0.01)"
  module: cst.Module = cst.parse_module(code)
  call_node: cst.Call = cast(cst.Call, module.body[0].body[0].value)

  transformed: cst.Call = transform_mlx_optimizer_init(call_node, ctx)
  assert transformed.args[0].keyword is not None
  assert transformed.args[0].keyword.value == "learning_rate"


def test_transform_optimizer_init_lr_rename_no_keyword_true_actually() -> None:
  """Docstring."""
  ctx: MagicMock = MagicMock()
  ctx.current_op_id = "Adam"
  ctx.lookup_api.return_value = "mlx.optimizers.Adam"
  code: str = "optim.Adam(lr=0.01)"
  module: cst.Module = cst.parse_module(code)
  call_node: cst.Call = cast(cst.Call, module.body[0].body[0].value)
  transformed: cst.Call = transform_mlx_optimizer_init(call_node, ctx)
  assert transformed.args[0].keyword is not None
  assert transformed.args[0].keyword.value == "learning_rate"
