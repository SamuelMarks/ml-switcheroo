import pytest
import libcst as cst
from unittest.mock import MagicMock


def get_dummy_ctx(target_fw="torch", current_op_id="dummy", sharding_supported=False):
  ctx = MagicMock()
  ctx.target_fw = target_fw
  ctx.current_op_id = current_op_id
  op_def = MagicMock()
  op_def.sharding_supported = sharding_supported
  ctx.semantics.get_operation.return_value = op_def
  ctx.lookup_api.return_value = "dummy.method"
  return ctx


def test_plugin_coverage_gaps():
  from ml_switcheroo.plugins.auto_fsdp_wrapper import wrap_with_sharding

  node = cst.Call(func=cst.Name("dummy"), args=[])
  wrap_with_sharding(node, get_dummy_ctx(target_fw="unknown", sharding_supported=True))

  from ml_switcheroo.plugins.casting import transform_casting

  ctx = get_dummy_ctx()
  ctx.semantics.get_operation.return_value = None
  transform_casting(node, ctx)
  ctx.semantics.get_operation.return_value = MagicMock(returns=[MagicMock(type="unknown_type")])
  transform_casting(node, ctx)

  from ml_switcheroo.plugins.clipping import transform_grad_clipping

  ctx = get_dummy_ctx()
  ctx.semantics.get_operation.return_value = MagicMock(grad_clip_value=None)
  transform_grad_clipping(node, ctx)

  from ml_switcheroo.plugins.io_handler import transform_io_calls

  transform_io_calls(node, get_dummy_ctx(target_fw="unknown"))

  from ml_switcheroo.plugins.keras_sequential import transform_keras_sequential

  transform_keras_sequential(node, get_dummy_ctx(target_fw="unknown"))

  from ml_switcheroo.plugins.method_property import transform_method_to_property

  ctx = get_dummy_ctx(target_fw="unknown")
  ctx.semantics.get_operation.return_value = MagicMock(is_property=True)
  transform_method_to_property(node, ctx)

  from ml_switcheroo.plugins.nnx_to_torch_params import transform_nnx_param

  transform_nnx_param(node, get_dummy_ctx(target_fw="unknown"))

  from ml_switcheroo.plugins.scatter import transform_scatter

  ctx = get_dummy_ctx(target_fw="unknown")
  ctx.semantics.get_operation.return_value = MagicMock(is_scatter=True)
  transform_scatter(node, ctx)

  from ml_switcheroo.plugins.state_flag_injection import inject_training_flag_call, capture_eval_state

  inject_training_flag_call(node, get_dummy_ctx(target_fw="unknown"))
  capture_eval_state(node, get_dummy_ctx(target_fw="unknown"))

  from ml_switcheroo.plugins.static_unroll import unroll_static_loops

  unroll_static_loops(
    cst.For(target=cst.Name("i"), iter=cst.Name("a"), body=cst.IndentedBlock(body=[])), get_dummy_ctx(target_fw="unknown")
  )

  from ml_switcheroo.plugins.tf_data_loader import transform_tf_dataloader

  ctx = get_dummy_ctx(target_fw="unknown")
  ctx.semantics.get_operation.return_value = MagicMock(is_tf_data=True)
  transform_tf_dataloader(node, ctx)
