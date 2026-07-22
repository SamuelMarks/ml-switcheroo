"""Test suite for the Nnx To Torch Params module."""

import libcst as cst
from unittest.mock import MagicMock
from ml_switcheroo.plugins.nnx_to_torch_params import transform_nnx_param
from ml_switcheroo.core.hooks import HookContext


def test_nnx_param_trainable():
  """Verifies the behavior of NNX parameter trainable."""
  node = cst.Call(
    func=cst.Attribute(value=cst.Name("nnx"), attr=cst.Name("Param")), args=[cst.Arg(value=cst.Name("zeros"))]
  )
  ctx = HookContext(semantics=MagicMock(), config=MagicMock())
  ctx.current_op_id = "Param"
  ctx.lookup_api = MagicMock(return_value="torch.nn.Parameter")
  res = transform_nnx_param(node, ctx)
  assert res.func.value.value.value == "torch"
  assert res.func.attr.value == "Parameter"
  assert len(res.args) == 1


def test_nnx_param_batch_stat():
  """Verifies the behavior of NNX parameter batch statistic."""
  node = cst.Call(
    func=cst.Attribute(value=cst.Name("nnx"), attr=cst.Name("BatchStat")), args=[cst.Arg(value=cst.Name("zeros"))]
  )
  ctx = HookContext(semantics=MagicMock(), config=MagicMock())
  ctx.current_op_id = "BatchStat"
  ctx.lookup_api = MagicMock(return_value="torch.nn.Parameter")
  res = transform_nnx_param(node, ctx)
  assert len(res.args) == 2
  assert res.args[1].keyword.value == "requires_grad"
  assert res.args[1].value.value == "False"


def test_nnx_param_batch_stat_already_has_requires_grad():
  """Verifies the behavior of NNX parameter batch statistic already has requires grad."""
  node = cst.Call(
    func=cst.Attribute(value=cst.Name("nnx"), attr=cst.Name("BatchStat")),
    args=[cst.Arg(value=cst.Name("zeros")), cst.Arg(keyword=cst.Name("requires_grad"), value=cst.Name("False"))],
  )
  ctx = HookContext(semantics=MagicMock(), config=MagicMock())
  ctx.current_op_id = "BatchStat"
  ctx.lookup_api = MagicMock(return_value="torch.nn.Parameter")
  res = transform_nnx_param(node, ctx)
  assert len(res.args) == 2


def test_nnx_param_missing_api():
  """Verifies the behavior of NNX parameter missing API."""
  node = cst.Call(func=cst.Name("Param"))
  ctx = HookContext(semantics=MagicMock(), config=MagicMock())
  ctx.lookup_api = MagicMock(return_value=None)
  res = transform_nnx_param(node, ctx)
  assert res is node


def test_nnx_param_leaf_name_fallback():
  """Verifies the behavior of NNX parameter leaf name fallback."""
  node = cst.Call(func=cst.Name("Unknown"))
  ctx = HookContext(semantics=MagicMock(), config=MagicMock())
  ctx.lookup_api = MagicMock(return_value="torch.nn.Parameter")
  res = transform_nnx_param(node, ctx)
  assert len(res.args) == 0


def test_nnx_param_batch_stat_no_args():
  """Verifies the behavior of NNX parameter batch statistic no arguments."""
  node = cst.Call(func=cst.Attribute(value=cst.Name("nnx"), attr=cst.Name("BatchStat")))
  ctx = HookContext(semantics=MagicMock(), config=MagicMock())
  ctx.current_op_id = "BatchStat"
  ctx.lookup_api = MagicMock(return_value="torch.nn.Parameter")
  res = transform_nnx_param(node, ctx)
  assert len(res.args) == 1
  assert res.args[0].keyword.value == "requires_grad"


def test_nnx_param_batch_stat_explicit_comma():
  """Verifies the behavior of NNX parameter batch statistic explicit comma."""
  node = cst.Call(
    func=cst.Attribute(value=cst.Name("nnx"), attr=cst.Name("BatchStat")),
    args=[cst.Arg(value=cst.Name("zeros"), comma=cst.Comma())],
  )
  ctx = HookContext(semantics=MagicMock(), config=MagicMock())
  ctx.current_op_id = "BatchStat"
  ctx.lookup_api = MagicMock(return_value="torch.nn.Parameter")
  res = transform_nnx_param(node, ctx)
  assert len(res.args) == 2
