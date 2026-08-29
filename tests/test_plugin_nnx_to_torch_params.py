"""Docstring."""

from unittest.mock import MagicMock

import libcst as cst

from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.nnx_to_torch_params import _extract_leaf_name, transform_nnx_param


def test_extract_leaf_name_unsupported() -> None:
  """Docstring."""
  node: cst.Call = cst.Call(func=cst.Name("foo"), args=[])
  assert _extract_leaf_name(node) is None


def test_transform_nnx_param_missing_api() -> None:
  """Docstring."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.current_op_id = None
  ctx.lookup_api.return_value = None

  code: str = "nnx.Param(zeros(1))"
  module: cst.Module = cst.parse_module(code)
  call_node: cst.BaseExpression = module.body[0].body[0].value

  transformed: cst.CSTNode = transform_nnx_param(call_node, ctx)
  assert transformed is call_node


def test_transform_nnx_param_trainable() -> None:
  """Docstring."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.current_op_id = "Param"
  ctx.lookup_api.return_value = "torch.nn.Parameter"

  code: str = "nnx.Param(zeros(1))"
  module: cst.Module = cst.parse_module(code)
  call_node: cst.BaseExpression = module.body[0].body[0].value

  transformed: cst.CSTNode = transform_nnx_param(call_node, ctx)
  assert isinstance(transformed, cst.Call)
  assert isinstance(transformed.func, cst.Attribute)
  assert transformed.func.attr.value == "Parameter"
  assert len(transformed.args) == 1


def test_transform_nnx_param_non_trainable_injects_requires_grad() -> None:
  """Docstring."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.current_op_id = "BatchStat"
  ctx.lookup_api.return_value = "torch.nn.Parameter"

  code: str = "nnx.BatchStat(zeros(1))"
  module: cst.Module = cst.parse_module(code)
  call_node: cst.BaseExpression = module.body[0].body[0].value

  transformed: cst.CSTNode = transform_nnx_param(call_node, ctx)
  assert isinstance(transformed, cst.Call)
  assert len(transformed.args) == 2
  assert getattr(transformed.args[1].keyword, "value", None) == "requires_grad"
  assert getattr(transformed.args[1].value, "value", None) == "False"


def test_transform_nnx_param_non_trainable_has_requires_grad() -> None:
  """Docstring."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.current_op_id = "BatchStat"
  ctx.lookup_api.return_value = "torch.nn.Parameter"

  code: str = "nnx.BatchStat(zeros(1), requires_grad=True)"
  module: cst.Module = cst.parse_module(code)
  call_node: cst.BaseExpression = module.body[0].body[0].value

  transformed: cst.CSTNode = transform_nnx_param(call_node, ctx)
  assert isinstance(transformed, cst.Call)
  assert len(transformed.args) == 2
  assert getattr(transformed.args[1].keyword, "value", None) == "requires_grad"
  assert getattr(transformed.args[1].value, "value", None) == "True"


def test_transform_nnx_param_with_no_args_non_trainable() -> None:
  """Docstring."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.current_op_id = "BatchStat"
  ctx.lookup_api.return_value = "torch.nn.Parameter"

  code: str = "nnx.BatchStat()"
  module: cst.Module = cst.parse_module(code)
  call_node: cst.BaseExpression = module.body[0].body[0].value

  transformed: cst.CSTNode = transform_nnx_param(call_node, ctx)
  assert isinstance(transformed, cst.Call)
  assert len(transformed.args) == 1
  assert getattr(transformed.args[0].keyword, "value", None) == "requires_grad"
