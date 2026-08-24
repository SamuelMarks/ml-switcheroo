"""Docstring."""

import libcst as cst
from unittest.mock import MagicMock
from ml_switcheroo.plugins.auto_fsdp_wrapper import wrap_with_sharding
from ml_switcheroo.core.hooks import HookContext


def test_wrap_with_sharding_no_op_def():
  """Docstring."""
  node = cst.Call(func=cst.Name("Linear"))
  ctx = MagicMock(spec=HookContext)
  ctx.current_op_id = None
  result = wrap_with_sharding(node, ctx)
  assert result is node


def test_wrap_with_sharding_no_sharding_support():
  """Docstring."""
  node = cst.Call(func=cst.Name("Linear"))
  ctx = MagicMock(spec=HookContext)
  ctx.current_op_id = "nn.Linear"
  op_def = MagicMock()
  op_def.sharding_supported = False
  ctx.semantics = MagicMock()
  ctx.semantics.get_operation.return_value = op_def
  result = wrap_with_sharding(node, ctx)
  assert result is node


def test_wrap_with_sharding_no_api():
  """Docstring."""
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


def test_wrap_with_sharding_fsdp():
  """Docstring."""
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
  assert result.func.attr.value == "FSDP"
  assert len(result.args) == 2
  assert result.args[1].keyword.value == "use_orig_params"


def test_wrap_with_sharding_pjit():
  """Docstring."""
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
  assert result.func.attr.value == "pjit"
  assert len(result.args) == 1


def test_wrap_with_sharding_other():
  """Docstring."""
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
