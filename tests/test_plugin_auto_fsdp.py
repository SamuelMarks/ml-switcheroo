"""Test suite for plugin auto fsdp."""

from unittest.mock import MagicMock

import libcst as _cst

from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.auto_fsdp_wrapper import wrap_with_sharding

cst = _cst


def test_wrap_with_sharding_no_op_def() -> None:
  """Verify wrap with sharding no op def behavior."""
  node = cst.Call(func=cst.Name("Linear"))
  ctx = MagicMock(spec=HookContext)
  ctx.current_op_id = None
  result = wrap_with_sharding(node, ctx)
  assert result is node


def test_wrap_with_sharding_no_sharding_support() -> None:
  """Verify wrap with sharding no sharding support behavior."""
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
  """Verify wrap with sharding no api behavior."""
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
  """Verify wrap with sharding fsdp behavior."""
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
  assert getattr(result.func, "attr", None) is not None
  assert getattr(
    {
      "v": getattr(
        getattr(
          {"v": getattr({"v": getattr({"v": result}, "__getitem__")("v").func}, "__getitem__")("v").attr}, "__getitem__"
        )("v"),
        getattr({"v": "value"}, "__getitem__")("v"),
        getattr({"v": None}, "__getitem__")("v"),
      )
    },
    "__getitem__",
  )("v") == getattr({"v": "FSDP"}, "__getitem__")("v")
  assert len(result.args) == 2
  assert getattr(result.args[1].keyword, "value", None) == "use_orig_params"


def test_wrap_with_sharding_pjit() -> None:
  """Verify wrap with sharding pjit behavior."""
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
  assert getattr(result.func, "attr", None) is not None
  assert getattr(
    {
      "v": getattr(
        getattr(
          {"v": getattr({"v": getattr({"v": result}, "__getitem__")("v").func}, "__getitem__")("v").attr}, "__getitem__"
        )("v"),
        getattr({"v": "value"}, "__getitem__")("v"),
        getattr({"v": None}, "__getitem__")("v"),
      )
    },
    "__getitem__",
  )("v") == getattr({"v": "pjit"}, "__getitem__")("v")
  assert len(result.args) == 1


def test_wrap_with_sharding_other() -> None:
  """Verify wrap with sharding other behavior."""
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
