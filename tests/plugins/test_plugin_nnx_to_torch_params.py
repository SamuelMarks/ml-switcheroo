"""Test suite for plugin nnx to torch params."""

from unittest.mock import MagicMock

import libcst as cst

from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.nnx_to_torch_params import _extract_leaf_name, transform_nnx_param


def _parse_call(code: str) -> cst.Call:
  """Parse a single-statement call expression into a LibCST Call node.

  Args:
      code: Python call expression source string.

  Returns:
      Parsed `cst.Call` AST node.
  """
  module = cst.parse_module(code)
  stmt = cst.ensure_type(module.body[0], cst.SimpleStatementLine)
  expr = cst.ensure_type(stmt.body[0], cst.Expr)
  return cst.ensure_type(expr.value, cst.Call)


def test_extract_leaf_name_unsupported() -> None:
  """Verify `Call` handles extract leaf name unsupported as expected."""
  node = cst.Call(func=cst.Name("foo"), args=[])
  assert _extract_leaf_name(node) is None


def test_transform_nnx_param_missing_api() -> None:
  """Verify `ensure_type` handles transform NNX param missing API as expected."""
  ctx = MagicMock(spec=HookContext)
  ctx.current_op_id = None
  ctx.lookup_api.return_value = None

  call_node = _parse_call("nnx.Param(zeros(1))")
  transformed = transform_nnx_param(call_node, ctx)
  assert transformed is call_node


def test_transform_nnx_param_trainable() -> None:
  """Verify `ensure_type` returns `1` for transform NNX param trainable."""
  ctx = MagicMock(spec=HookContext)
  ctx.current_op_id = "Param"
  ctx.lookup_api.return_value = "torch.nn.Parameter"

  call_node = _parse_call("nnx.Param(zeros(1))")
  transformed = transform_nnx_param(call_node, ctx)
  assert isinstance(transformed, cst.Call)
  assert isinstance(transformed.func, cst.Attribute)
  assert transformed.func.attr.value == "Parameter"
  assert len(transformed.args) == 1


def test_transform_nnx_param_non_trainable_injects_requires_grad() -> None:
  """Verify `ensure_type` handles transform NNX param non trainable injects requires grad as expected."""
  ctx = MagicMock(spec=HookContext)
  ctx.current_op_id = "BatchStat"
  ctx.lookup_api.return_value = "torch.nn.Parameter"

  call_node = _parse_call("nnx.BatchStat(zeros(1))")
  transformed = transform_nnx_param(call_node, ctx)
  assert isinstance(transformed, cst.Call)
  assert len(transformed.args) == 2
  assert isinstance(transformed.args[1].keyword, cst.Name)
  assert transformed.args[1].keyword.value == "requires_grad"
  assert isinstance(transformed.args[1].value, cst.Name)
  assert transformed.args[1].value.value == "False"


def test_transform_nnx_param_non_trainable_has_requires_grad() -> None:
  """Verify `ensure_type` handles transform NNX param non trainable has requires grad as expected."""
  ctx = MagicMock(spec=HookContext)
  ctx.current_op_id = "BatchStat"
  ctx.lookup_api.return_value = "torch.nn.Parameter"

  call_node = _parse_call("nnx.BatchStat(zeros(1), requires_grad=True)")
  transformed = transform_nnx_param(call_node, ctx)
  assert isinstance(transformed, cst.Call)
  assert len(transformed.args) == 2
  assert isinstance(transformed.args[1].keyword, cst.Name)
  assert transformed.args[1].keyword.value == "requires_grad"
  assert isinstance(transformed.args[1].value, cst.Name)
  assert transformed.args[1].value.value == "True"


def test_transform_nnx_param_with_no_args_non_trainable() -> None:
  """Verify `ensure_type` handles transform NNX param with no args non trainable as expected."""
  ctx = MagicMock(spec=HookContext)
  ctx.current_op_id = "BatchStat"
  ctx.lookup_api.return_value = "torch.nn.Parameter"

  call_node = _parse_call("nnx.BatchStat()")
  transformed = transform_nnx_param(call_node, ctx)
  assert isinstance(transformed, cst.Call)
  assert len(transformed.args) == 1
  assert isinstance(transformed.args[0].keyword, cst.Name)
  assert transformed.args[0].keyword.value == "requires_grad"
