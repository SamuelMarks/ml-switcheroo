"""Test suite for the Nnx To Torch Params module."""

from typing import Union
from unittest.mock import MagicMock

import libcst as cst

from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.nnx_to_torch_params import transform_nnx_param


def test_nnx_param_trainable() -> None:
  """Verifies the behavior of NNX parameter trainable."""
  node: cst.Call = cst.Call(
    func=cst.Attribute(value=cst.Name("nnx"), attr=cst.Name("Param")), args=[cst.Arg(value=cst.Name("zeros"))]
  )
  ctx: HookContext = HookContext(semantics=MagicMock(), config=MagicMock())
  ctx.current_op_id = "Param"
  ctx.lookup_api = MagicMock(return_value="torch.nn.Parameter")
  res: Union[cst.CSTNode, cst.Call] = transform_nnx_param(node, ctx)
  assert isinstance(res, cst.Call)
  assert isinstance(res.func, cst.Attribute)
  assert isinstance(res.func.value, cst.Attribute)
  assert isinstance(res.func.value.value, cst.Name)
  assert res.func.value.value.value == "torch"
  assert res.func.attr.value == "Parameter"
  assert len(res.args) == 1


def test_nnx_param_batch_stat() -> None:
  """Verifies the behavior of NNX parameter batch statistic."""
  node: cst.Call = cst.Call(
    func=cst.Attribute(value=cst.Name("nnx"), attr=cst.Name("BatchStat")), args=[cst.Arg(value=cst.Name("zeros"))]
  )
  ctx: HookContext = HookContext(semantics=MagicMock(), config=MagicMock())
  ctx.current_op_id = "BatchStat"
  ctx.lookup_api = MagicMock(return_value="torch.nn.Parameter")
  res: Union[cst.CSTNode, cst.Call] = transform_nnx_param(node, ctx)
  assert isinstance(res, cst.Call)
  assert len(res.args) == 2
  assert res.args[1].keyword is not None
  assert res.args[1].keyword.value == "requires_grad"
  assert isinstance(res.args[1].value, cst.Name)
  assert res.args[1].value.value == "False"


def test_nnx_param_batch_stat_already_has_requires_grad() -> None:
  """Verifies the behavior of NNX parameter batch statistic already has requires grad."""
  node: cst.Call = cst.Call(
    func=cst.Attribute(value=cst.Name("nnx"), attr=cst.Name("BatchStat")),
    args=[cst.Arg(value=cst.Name("zeros")), cst.Arg(keyword=cst.Name("requires_grad"), value=cst.Name("False"))],
  )
  ctx: HookContext = HookContext(semantics=MagicMock(), config=MagicMock())
  ctx.current_op_id = "BatchStat"
  ctx.lookup_api = MagicMock(return_value="torch.nn.Parameter")
  res: Union[cst.CSTNode, cst.Call] = transform_nnx_param(node, ctx)
  assert isinstance(res, cst.Call)
  assert len(res.args) == 2


def test_nnx_param_missing_api() -> None:
  """Verifies the behavior of NNX parameter missing API."""
  node: cst.Call = cst.Call(func=cst.Name("Param"))
  ctx: HookContext = HookContext(semantics=MagicMock(), config=MagicMock())
  ctx.lookup_api = MagicMock(return_value=None)
  res: Union[cst.CSTNode, cst.Call] = transform_nnx_param(node, ctx)
  assert res is node


def test_nnx_param_leaf_name_fallback() -> None:
  """Verifies the behavior of NNX parameter leaf name fallback."""
  node: cst.Call = cst.Call(func=cst.Name("Unknown"))
  ctx: HookContext = HookContext(semantics=MagicMock(), config=MagicMock())
  ctx.lookup_api = MagicMock(return_value="torch.nn.Parameter")
  res: Union[cst.CSTNode, cst.Call] = transform_nnx_param(node, ctx)
  assert isinstance(res, cst.Call)
  assert len(res.args) == 0


def test_nnx_param_unsupported_type() -> None:
  """Verifies that an unsupported func node type returns None from _get_leaf_name."""
  node: cst.Call = cst.Call(func=cst.SimpleString("'string'"))
  ctx: HookContext = HookContext(semantics=MagicMock(), config=MagicMock())
  ctx.lookup_api = MagicMock(return_value="torch.nn.Parameter")
  res: Union[cst.CSTNode, cst.Call] = transform_nnx_param(node, ctx)
  assert isinstance(res, cst.Call)
  assert isinstance(res.func, cst.Attribute)
  assert isinstance(res.func.value, cst.Attribute)
  assert isinstance(res.func.value.value, cst.Name)
  assert res.func.value.value.value == "torch"
  assert res.func.attr.value == "Parameter"


def test_nnx_param_batch_stat_no_args() -> None:
  """Verifies the behavior of NNX parameter batch statistic no arguments."""
  node: cst.Call = cst.Call(func=cst.Attribute(value=cst.Name("nnx"), attr=cst.Name("BatchStat")))
  ctx: HookContext = HookContext(semantics=MagicMock(), config=MagicMock())
  ctx.current_op_id = "BatchStat"
  ctx.lookup_api = MagicMock(return_value="torch.nn.Parameter")
  res: Union[cst.CSTNode, cst.Call] = transform_nnx_param(node, ctx)
  assert isinstance(res, cst.Call)
  assert len(res.args) == 1
  assert res.args[0].keyword is not None
  assert res.args[0].keyword.value == "requires_grad"


def test_nnx_param_batch_stat_explicit_comma() -> None:
  """Verifies the behavior of NNX parameter batch statistic explicit comma."""
  node: cst.Call = cst.Call(
    func=cst.Attribute(value=cst.Name("nnx"), attr=cst.Name("BatchStat")),
    args=[cst.Arg(value=cst.Name("zeros"), comma=cst.Comma())],
  )
  ctx: HookContext = HookContext(semantics=MagicMock(), config=MagicMock())
  ctx.current_op_id = "BatchStat"
  ctx.lookup_api = MagicMock(return_value="torch.nn.Parameter")
  res: Union[cst.CSTNode, cst.Call] = transform_nnx_param(node, ctx)
  assert isinstance(res, cst.Call)
  assert len(res.args) == 2
