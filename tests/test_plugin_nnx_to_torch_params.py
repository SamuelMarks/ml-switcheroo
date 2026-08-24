"""Docstring."""

import libcst as cst
from unittest.mock import MagicMock

from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.nnx_to_torch_params import _extract_leaf_name, transform_nnx_param


def test_extract_leaf_name_unsupported():
  """Docstring."""
  node = cst.Call(func=cst.Name("foo"), args=[])
  assert _extract_leaf_name(node) is None


def test_transform_nnx_param_missing_api():
  """Docstring."""
  ctx = MagicMock(spec=HookContext)
  ctx.current_op_id = None
  ctx.lookup_api.return_value = None

  code = "nnx.Param(zeros(1))"
  module = cst.parse_module(code)
  call_node = module.body[0].body[0].value

  transformed = transform_nnx_param(call_node, ctx)
  assert transformed is call_node


def test_transform_nnx_param_trainable():
  """Docstring."""
  ctx = MagicMock(spec=HookContext)
  ctx.current_op_id = "Param"
  ctx.lookup_api.return_value = "torch.nn.Parameter"

  code = "nnx.Param(zeros(1))"
  module = cst.parse_module(code)
  call_node = module.body[0].body[0].value

  transformed = transform_nnx_param(call_node, ctx)
  assert isinstance(transformed.func, cst.Attribute)
  assert transformed.func.attr.value == "Parameter"
  assert len(transformed.args) == 1


def test_transform_nnx_param_non_trainable_injects_requires_grad():
  """Docstring."""
  ctx = MagicMock(spec=HookContext)
  ctx.current_op_id = "BatchStat"
  ctx.lookup_api.return_value = "torch.nn.Parameter"

  code = "nnx.BatchStat(zeros(1))"
  module = cst.parse_module(code)
  call_node = module.body[0].body[0].value

  transformed = transform_nnx_param(call_node, ctx)
  assert len(transformed.args) == 2
  assert transformed.args[1].keyword.value == "requires_grad"
  assert transformed.args[1].value.value == "False"


def test_transform_nnx_param_non_trainable_has_requires_grad():
  """Docstring."""
  ctx = MagicMock(spec=HookContext)
  ctx.current_op_id = "BatchStat"
  ctx.lookup_api.return_value = "torch.nn.Parameter"

  code = "nnx.BatchStat(zeros(1), requires_grad=True)"
  module = cst.parse_module(code)
  call_node = module.body[0].body[0].value

  transformed = transform_nnx_param(call_node, ctx)
  assert len(transformed.args) == 2
  assert transformed.args[1].keyword.value == "requires_grad"
  assert transformed.args[1].value.value == "True"


def test_transform_nnx_param_with_no_args_non_trainable():
  """Docstring."""
  ctx = MagicMock(spec=HookContext)
  ctx.current_op_id = "BatchStat"
  ctx.lookup_api.return_value = "torch.nn.Parameter"

  code = "nnx.BatchStat()"
  module = cst.parse_module(code)
  call_node = module.body[0].body[0].value

  transformed = transform_nnx_param(call_node, ctx)
  assert len(transformed.args) == 1
  assert transformed.args[0].keyword.value == "requires_grad"
