"""Test suite for plugin auto fsdp."""

import typing

from unittest.mock import MagicMock

import libcst as _cst

from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.auto_fsdp_wrapper import wrap_with_sharding

cst: typing.Any = _cst


def test_wrap_with_sharding_no_op_def() -> None:
  """Verify wrap with sharding no op def behavior."""
  node: typing.Any = cst.Call(func=cst.Name("Linear"))
  ctx: typing.Any = MagicMock(spec=HookContext)
  ctx.current_op_id = None
  result: typing.Any = wrap_with_sharding(node, ctx)
  assert result is node


def test_wrap_with_sharding_no_sharding_support() -> None:
  """Verify wrap with sharding no sharding support behavior."""
  node: typing.Any = cst.Call(func=cst.Name("Linear"))
  ctx: typing.Any = MagicMock(spec=HookContext)
  ctx.current_op_id = "nn.Linear"
  op_def: typing.Any = MagicMock()
  op_def.sharding_supported = False
  ctx.semantics = MagicMock()
  ctx.semantics.get_operation.return_value = op_def
  result: typing.Any = wrap_with_sharding(node, ctx)
  assert result is node


def test_wrap_with_sharding_no_api() -> None:
  """Verify wrap with sharding no api behavior."""
  node: typing.Any = cst.Call(func=cst.Name("Linear"))
  ctx: typing.Any = MagicMock(spec=HookContext)
  ctx.current_op_id = "nn.Linear"
  op_def: typing.Any = MagicMock()
  op_def.sharding_supported = True
  ctx.semantics = MagicMock()
  ctx.semantics.get_operation.return_value = op_def
  ctx.plugin_traits.sharding_wrapper_api = None
  result: typing.Any = wrap_with_sharding(node, ctx)
  assert result is node


def test_wrap_with_sharding_fsdp() -> None:
  """Verify wrap with sharding fsdp behavior."""
  node: typing.Any = cst.Call(func=cst.Name("Linear"))
  ctx: typing.Any = MagicMock(spec=HookContext)
  ctx.current_op_id = "nn.Linear"
  op_def: typing.Any = MagicMock()
  op_def.sharding_supported = True
  ctx.semantics = MagicMock()
  ctx.semantics.get_operation.return_value = op_def
  ctx.plugin_traits.sharding_wrapper_api = "torch.distributed.fsdp.FSDP"
  result: typing.Any = wrap_with_sharding(node, ctx)
  assert isinstance(result, cst.Call)
  assert getattr(result.func, "attr", None) is not None
  assert getattr(result.func.attr, "value", None) == "FSDP"
  assert len(result.args) == 2
  assert getattr(result.args[1].keyword, "value", None) == "use_orig_params"


def test_wrap_with_sharding_pjit() -> None:
  """Verify wrap with sharding pjit behavior."""
  node: typing.Any = cst.Call(func=cst.Name("Linear"))
  ctx: typing.Any = MagicMock(spec=HookContext)
  ctx.current_op_id = "nn.Linear"
  op_def: typing.Any = MagicMock()
  op_def.sharding_supported = True
  ctx.semantics = MagicMock()
  ctx.semantics.get_operation.return_value = op_def
  ctx.plugin_traits.sharding_wrapper_api = "jax.experimental.pjit.pjit"
  result: typing.Any = wrap_with_sharding(node, ctx)
  assert isinstance(result, cst.Call)
  assert getattr(result.func, "attr", None) is not None
  assert getattr(result.func.attr, "value", None) == "pjit"
  assert len(result.args) == 1


def test_wrap_with_sharding_other() -> None:
  """Verify wrap with sharding other behavior."""
  node: typing.Any = cst.Call(func=cst.Name("Linear"))
  ctx: typing.Any = MagicMock(spec=HookContext)
  ctx.current_op_id = "nn.Linear"
  op_def: typing.Any = MagicMock()
  op_def.sharding_supported = True
  ctx.semantics = MagicMock()
  ctx.semantics.get_operation.return_value = op_def
  ctx.plugin_traits.sharding_wrapper_api = "some.other.wrapper"
  result: typing.Any = wrap_with_sharding(node, ctx)
  assert result is node
