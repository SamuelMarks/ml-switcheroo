"""Docstring."""

import libcst as cst
from unittest.mock import MagicMock
from ml_switcheroo.plugins.auto_fsdp_wrapper import wrap_with_sharding
from ml_switcheroo.core.hooks import HookContext


def test_wrap_with_sharding_no_op_def() -> None:
  """Docstring."""
  node: cst.Call = cst.Call(func=cst.Name("Linear"))
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.current_op_id = None
  result: cst.CSTNode = wrap_with_sharding(node, ctx)
  assert result is node


def test_wrap_with_sharding_no_sharding_support() -> None:
  """Docstring."""
  node: cst.Call = cst.Call(func=cst.Name("Linear"))
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.current_op_id = "nn.Linear"
  op_def: MagicMock = MagicMock()
  op_def.sharding_supported = False
  ctx.semantics = MagicMock()
  ctx.semantics.get_operation.return_value = op_def
  result: cst.CSTNode = wrap_with_sharding(node, ctx)
  assert result is node


def test_wrap_with_sharding_no_api() -> None:
  """Docstring."""
  node: cst.Call = cst.Call(func=cst.Name("Linear"))
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.current_op_id = "nn.Linear"
  op_def: MagicMock = MagicMock()
  op_def.sharding_supported = True
  ctx.semantics = MagicMock()
  ctx.semantics.get_operation.return_value = op_def
  ctx.plugin_traits.sharding_wrapper_api = None
  result: cst.CSTNode = wrap_with_sharding(node, ctx)
  assert result is node


def test_wrap_with_sharding_fsdp() -> None:
  """Docstring."""
  node: cst.Call = cst.Call(func=cst.Name("Linear"))
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.current_op_id = "nn.Linear"
  op_def: MagicMock = MagicMock()
  op_def.sharding_supported = True
  ctx.semantics = MagicMock()
  ctx.semantics.get_operation.return_value = op_def
  ctx.plugin_traits.sharding_wrapper_api = "torch.distributed.fsdp.FSDP"
  result: cst.CSTNode = wrap_with_sharding(node, ctx)
  assert isinstance(result, cst.Call)
  assert getattr(result.func, "attr", None) is not None
  assert getattr(result.func.attr, "value", None) == "FSDP"
  assert len(result.args) == 2
  assert getattr(result.args[1].keyword, "value", None) == "use_orig_params"


def test_wrap_with_sharding_pjit() -> None:
  """Docstring."""
  node: cst.Call = cst.Call(func=cst.Name("Linear"))
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.current_op_id = "nn.Linear"
  op_def: MagicMock = MagicMock()
  op_def.sharding_supported = True
  ctx.semantics = MagicMock()
  ctx.semantics.get_operation.return_value = op_def
  ctx.plugin_traits.sharding_wrapper_api = "jax.experimental.pjit.pjit"
  result: cst.CSTNode = wrap_with_sharding(node, ctx)
  assert isinstance(result, cst.Call)
  assert getattr(result.func, "attr", None) is not None
  assert getattr(result.func.attr, "value", None) == "pjit"
  assert len(result.args) == 1


def test_wrap_with_sharding_other() -> None:
  """Docstring."""
  node: cst.Call = cst.Call(func=cst.Name("Linear"))
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.current_op_id = "nn.Linear"
  op_def: MagicMock = MagicMock()
  op_def.sharding_supported = True
  ctx.semantics = MagicMock()
  ctx.semantics.get_operation.return_value = op_def
  ctx.plugin_traits.sharding_wrapper_api = "some.other.wrapper"
  result: cst.CSTNode = wrap_with_sharding(node, ctx)
  assert result is node
