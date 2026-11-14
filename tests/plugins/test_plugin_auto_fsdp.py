"""Test suite for plugin auto fsdp."""

from unittest.mock import MagicMock

import libcst as cst

from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.auto_fsdp_wrapper import wrap_with_sharding


def test_wrap_with_sharding_no_op_def() -> None:
  """Verify `Call` handles wrap with sharding no operation def as expected."""
  node = cst.Call(func=cst.Name("Linear"))
  ctx = MagicMock(spec=HookContext)
  ctx.current_op_id = None
  result = wrap_with_sharding(node, ctx)
  assert result is node


def test_wrap_with_sharding_no_sharding_support() -> None:
  """Verify `Call` handles wrap with sharding no sharding support as expected."""
  node = cst.Call(func=cst.Name("Linear"))
  ctx = MagicMock(spec=HookContext)
  ctx.current_op_id = "nn.Linear"
  op_def = MagicMock()
  op_def.sharding_supported = False
  ctx.semantics = MagicMock()
  ctx.semantics.get_operation.return_value = op_def
  result = wrap_with_sharding(node, ctx)
  assert result is node


def test_wrap_with_sharding_no_api() -> None:
  """Verify `Call` handles wrap with sharding no API as expected."""
  node = cst.Call(func=cst.Name("Linear"))
  ctx = MagicMock(spec=HookContext)
  ctx.current_op_id = "nn.Linear"
  op_def = MagicMock()
  op_def.sharding_supported = True
  ctx.semantics = MagicMock()
  ctx.semantics.get_operation.return_value = op_def
  ctx.plugin_traits.sharding_wrapper_api = None
  result = wrap_with_sharding(node, ctx)
  assert result is node


def test_wrap_with_sharding_fsdp() -> None:
  """Verify `Call` returns `2` for wrap with sharding FSDP."""
  node = cst.Call(func=cst.Name("Linear"))
  ctx = MagicMock(spec=HookContext)
  ctx.current_op_id = "nn.Linear"
  op_def = MagicMock()
  op_def.sharding_supported = True
  ctx.semantics = MagicMock()
  ctx.semantics.get_operation.return_value = op_def
  ctx.plugin_traits.sharding_wrapper_api = "torch.distributed.fsdp.FSDP"
  result = wrap_with_sharding(node, ctx)
  assert isinstance(result, cst.Call)
  assert isinstance(result.func, cst.Attribute)
  assert result.func.attr.value == "FSDP"
  assert len(result.args) == 2
  assert result.args[1].keyword is not None
  assert result.args[1].keyword.value == "use_orig_params"


def test_wrap_with_sharding_pjit() -> None:
  """Verify `Call` returns `1` for wrap with sharding pjit."""
  node = cst.Call(func=cst.Name("Linear"))
  ctx = MagicMock(spec=HookContext)
  ctx.current_op_id = "nn.Linear"
  op_def = MagicMock()
  op_def.sharding_supported = True
  ctx.semantics = MagicMock()
  ctx.semantics.get_operation.return_value = op_def
  ctx.plugin_traits.sharding_wrapper_api = "jax.experimental.pjit.pjit"
  result = wrap_with_sharding(node, ctx)
  assert isinstance(result, cst.Call)
  assert isinstance(result.func, cst.Attribute)
  assert result.func.attr.value == "pjit"
  assert len(result.args) == 1


def test_wrap_with_sharding_other() -> None:
  """Verify `Call` handles wrap with sharding other as expected."""
  node = cst.Call(func=cst.Name("Linear"))
  ctx = MagicMock(spec=HookContext)
  ctx.current_op_id = "nn.Linear"
  op_def = MagicMock()
  op_def.sharding_supported = True
  ctx.semantics = MagicMock()
  ctx.semantics.get_operation.return_value = op_def
  ctx.plugin_traits.sharding_wrapper_api = "some.other.wrapper"
  result = wrap_with_sharding(node, ctx)
  assert result is node
