import pytest
import libcst as cst
from unittest.mock import MagicMock
from ml_switcheroo.plugins.loss_wrapper import transform_loss_reduction


def get_dummy_ctx(target_fw="torch", current_op_id="dummy", sharding_supported=False):
  ctx = MagicMock()
  ctx.target_fw = target_fw
  ctx.current_op_id = current_op_id
  op_def = MagicMock()
  op_def.sharding_supported = sharding_supported
  ctx.semantics.get_operation.return_value = op_def
  return ctx


def test_plugin_loss_wrapper():
  ctx = get_dummy_ctx(target_fw="unknown")
  ctx.semantics.get_operation.return_value = MagicMock(is_loss=True)
  ctx.lookup_api.return_value = "dummy.loss"

  # 65-68: reduction is a variable
  node = cst.Call(func=cst.Name("dummy"), args=[cst.Arg(value=cst.Name("my_mode"), keyword=cst.Name("reduction"))])
  transform_loss_reduction(node, ctx)

  # 83: loss_op_id is empty
  ctx.current_op_id = ""
  ctx.lookup_api.return_value = "dummy.loss"
  transform_loss_reduction(node, ctx)

  # 120: wrapper_api is empty
  ctx.current_op_id = "CrossEntropyLoss"
  ctx.lookup_api.side_effect = lambda x: "dummy.loss" if x == "CrossEntropyLoss" else None
  node2 = cst.Call(func=cst.Name("dummy"), args=[cst.Arg(value=cst.SimpleString('"sum"'), keyword=cst.Name("reduction"))])
  transform_loss_reduction(node2, ctx)
