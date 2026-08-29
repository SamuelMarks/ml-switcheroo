"""Docstring."""

from unittest.mock import MagicMock

import libcst as cst

from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.tf_data_loader import _extract_tensor_dataset_inputs, _get_arg_by_name, transform_tf_dataloader


def parse_expr(code: str) -> cst.BaseExpression:
  """Docstring."""
  return cst.parse_expression(code)


def test_get_arg_by_name() -> None:
  """Docstring."""
  call: cst.BaseExpression = parse_expr("foo(a=1, b=2)")
  assert _get_arg_by_name(getattr(call, "args"), "b") is getattr(call, "args")[1]
  assert _get_arg_by_name(getattr(call, "args"), "c") is None


def test_extract_tensor_dataset_inputs() -> None:
  """Docstring."""
  # Name
  call1: cst.BaseExpression = parse_expr("TensorDataset(x, y)")
  assert _extract_tensor_dataset_inputs(call1) == [
    getattr(getattr(call1, "args")[0], "value"),
    getattr(getattr(call1, "args")[1], "value"),
  ]

  # Attribute
  call2: cst.BaseExpression = parse_expr("torch.utils.data.TensorDataset(x, y)")
  assert _extract_tensor_dataset_inputs(call2) == [
    getattr(getattr(call2, "args")[0], "value"),
    getattr(getattr(call2, "args")[1], "value"),
  ]

  # Not TensorDataset
  call3: cst.BaseExpression = parse_expr("Other(x, y)")
  assert _extract_tensor_dataset_inputs(call3) is None

  # Not a call
  assert _extract_tensor_dataset_inputs(parse_expr("x")) is None


def test_transform_tf_dataloader_empty_args() -> None:
  """Docstring."""
  call: cst.BaseExpression = parse_expr("DataLoader()")
  ctx: MagicMock = MagicMock(spec=HookContext)
  res: cst.BaseExpression = transform_tf_dataloader(call, ctx)
  assert res is call


def test_transform_tf_dataloader_with_tensor_dataset() -> None:
  """Docstring."""
  call: cst.BaseExpression = parse_expr("DataLoader(TensorDataset(x, y), batch_size=64, shuffle=True)")
  ctx: MagicMock = MagicMock(spec=HookContext)
  res: cst.BaseExpression = transform_tf_dataloader(call, ctx)
  code: str = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=res)])]).code
  assert "tf.data.Dataset.from_tensor_slices((x, y))" in code
  assert ".shuffle(buffer_size=1024)" in code
  assert ".batch(64)" in code
  assert ".prefetch(tf.data.AUTOTUNE)" in code


def test_transform_tf_dataloader_no_tensor_dataset_single_input_no_shuffle() -> None:
  """Docstring."""
  call: cst.BaseExpression = parse_expr("DataLoader(my_dataset)")
  ctx: MagicMock = MagicMock(spec=HookContext)
  res: cst.BaseExpression = transform_tf_dataloader(call, ctx)
  code: str = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=res)])]).code
  assert "tf.data.Dataset.from_tensor_slices(my_dataset)" in code
  assert ".shuffle" not in code
  assert ".batch(1)" in code
  assert ".prefetch(tf.data.AUTOTUNE)" in code
