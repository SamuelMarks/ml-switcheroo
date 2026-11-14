"""Test suite for plugin tf data loader."""

from unittest.mock import MagicMock

import libcst as cst

from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.tf_data_loader import _extract_tensor_dataset_inputs, _get_arg_by_name, transform_tf_dataloader


def parse_expr(code: str) -> cst.BaseExpression:
  """Execute `parse_expr` helper in plugin TensorFlow data loader tests.

  Args:
      code: Python or DSL source code string to parse or transpile.

  Returns:
      Resulting `cst.BaseExpression` from parse expr.
  """
  return cst.parse_expression(code)


def test_get_arg_by_name() -> None:
  """Verify `parse_expr` handles get arg by name as expected."""
  call: cst.Call = cst.ensure_type(parse_expr("foo(a=1, b=2)"), cst.Call)
  assert _get_arg_by_name(call.args, "b") is call.args[1]
  assert _get_arg_by_name(call.args, "c") is None


def test_extract_tensor_dataset_inputs() -> None:
  """Verify `parse_expr` handles extract tensor dataset inputs as expected."""
  # Name
  call1: cst.Call = cst.ensure_type(parse_expr("TensorDataset(x, y)"), cst.Call)
  assert _extract_tensor_dataset_inputs(call1) == [
    call1.args[0].value,
    call1.args[1].value,
  ]

  # Attribute
  call2: cst.Call = cst.ensure_type(parse_expr("torch.utils.data.TensorDataset(x, y)"), cst.Call)
  assert _extract_tensor_dataset_inputs(call2) == [
    call2.args[0].value,
    call2.args[1].value,
  ]

  # Not TensorDataset
  call3: cst.BaseExpression = parse_expr("Other(x, y)")
  assert _extract_tensor_dataset_inputs(call3) is None

  # Not a call
  assert _extract_tensor_dataset_inputs(parse_expr("x")) is None


def test_transform_tf_dataloader_empty_args() -> None:
  """Verify `ensure_type` handles transform TensorFlow dataloader empty args as expected."""
  call: cst.Call = cst.ensure_type(parse_expr("DataLoader()"), cst.Call)
  ctx: MagicMock = MagicMock(spec=HookContext)
  res: cst.CSTNode = transform_tf_dataloader(call, ctx)
  assert res is call


def test_transform_tf_dataloader_with_tensor_dataset() -> None:
  """Verify `ensure_type` handles transform TensorFlow dataloader with tensor dataset as expected."""
  call: cst.Call = cst.ensure_type(parse_expr("DataLoader(TensorDataset(x, y), batch_size=64, shuffle=True)"), cst.Call)
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = "tf.data.Dataset"
  res = transform_tf_dataloader(call, ctx)
  assert isinstance(res, cst.BaseExpression)
  code: str = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=res)])]).code
  assert "tf.data.Dataset.from_tensor_slices((x, y))" in code
  assert ".shuffle(buffer_size=1024)" in code
  assert ".batch(64)" in code
  assert ".prefetch(tf.data.AUTOTUNE)" in code


def test_transform_tf_dataloader_no_tensor_dataset_single_input_no_shuffle() -> None:
  """Verify transform TensorFlow dataloader no tensor dataset single input no shuffle works as expected."""
  call: cst.Call = cst.ensure_type(parse_expr("DataLoader(my_dataset)"), cst.Call)
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = "tf.data.Dataset"
  res = transform_tf_dataloader(call, ctx)
  assert isinstance(res, cst.BaseExpression)
  code: str = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=res)])]).code
  assert "tf.data.Dataset.from_tensor_slices(my_dataset)" in code
  assert ".shuffle" not in code
  assert ".batch(1)" in code
  assert ".prefetch(tf.data.AUTOTUNE)" in code
