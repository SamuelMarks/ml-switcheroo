"""Test suite for plugin tf data loader."""

from tests.conftest import _test_getattr

from unittest.mock import MagicMock

import libcst as cst

from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.tf_data_loader import _extract_tensor_dataset_inputs, _get_arg_by_name, transform_tf_dataloader


def parse_expr(code: str) -> cst.BaseExpression:
  """Execute parse expr helper logic.

  Args:
      code: Python or DSL source code string to parse or transpile.

  Returns:
      Resulting `cst.BaseExpression` from parse expr.
  """
  return cst.parse_expression(code)


def test_get_arg_by_name() -> None:
  """Verify get arg by name behavior."""
  call: cst.BaseExpression = parse_expr("foo(a=1, b=2)")
  assert _get_arg_by_name(_test_getattr(call, "args"), "b") is _test_getattr(call, "args")[1]
  assert _get_arg_by_name(_test_getattr(call, "args"), "c") is None


def test_extract_tensor_dataset_inputs() -> None:
  """Verify extract tensor dataset inputs behavior."""
  # Name
  call1: cst.BaseExpression = parse_expr("TensorDataset(x, y)")
  assert _extract_tensor_dataset_inputs(call1) == [
    _test_getattr(_test_getattr(call1, "args")[0], "value"),
    _test_getattr(_test_getattr(call1, "args")[1], "value"),
  ]

  # Attribute
  call2: cst.BaseExpression = parse_expr("torch.utils.data.TensorDataset(x, y)")
  assert _extract_tensor_dataset_inputs(call2) == [
    _test_getattr(_test_getattr(call2, "args")[0], "value"),
    _test_getattr(_test_getattr(call2, "args")[1], "value"),
  ]

  # Not TensorDataset
  call3: cst.BaseExpression = parse_expr("Other(x, y)")
  assert _extract_tensor_dataset_inputs(call3) is None

  # Not a call
  assert _extract_tensor_dataset_inputs(parse_expr("x")) is None


def test_transform_tf_dataloader_empty_args() -> None:
  """Verify transform tf dataloader empty args behavior."""
  call: cst.Call = cst.ensure_type(parse_expr("DataLoader()"), cst.Call)
  ctx: MagicMock = MagicMock(spec=HookContext)
  res: cst.BaseExpression = _test_getattr(
    {
      "v": _test_getattr({"v": transform_tf_dataloader}, "__getitem__")("v")(
        _test_getattr({"v": call}, "__getitem__")("v"), _test_getattr({"v": ctx}, "__getitem__")("v")
      )
    },
    "__getitem__",
  )("v")
  assert res is call


def test_transform_tf_dataloader_with_tensor_dataset() -> None:
  """Verify transform tf dataloader with tensor dataset behavior."""
  call: cst.Call = cst.ensure_type(parse_expr("DataLoader(TensorDataset(x, y), batch_size=64, shuffle=True)"), cst.Call)
  ctx: MagicMock = MagicMock(spec=HookContext)
  res: cst.BaseExpression = _test_getattr(
    {
      "v": _test_getattr({"v": transform_tf_dataloader}, "__getitem__")("v")(
        _test_getattr({"v": call}, "__getitem__")("v"), _test_getattr({"v": ctx}, "__getitem__")("v")
      )
    },
    "__getitem__",
  )("v")
  code: str = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=res)])]).code
  assert "tf.data.Dataset.from_tensor_slices((x, y))" in code
  assert ".shuffle(buffer_size=1024)" in code
  assert ".batch(64)" in code
  assert ".prefetch(tf.data.AUTOTUNE)" in code


def test_transform_tf_dataloader_no_tensor_dataset_single_input_no_shuffle() -> None:
  """Verify transform tf dataloader no tensor dataset single input no shuffle behavior."""
  call: cst.Call = cst.ensure_type(parse_expr("DataLoader(my_dataset)"), cst.Call)
  ctx: MagicMock = MagicMock(spec=HookContext)
  res: cst.BaseExpression = _test_getattr(
    {
      "v": _test_getattr({"v": transform_tf_dataloader}, "__getitem__")("v")(
        _test_getattr({"v": call}, "__getitem__")("v"), _test_getattr({"v": ctx}, "__getitem__")("v")
      )
    },
    "__getitem__",
  )("v")
  code: str = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=res)])]).code
  assert "tf.data.Dataset.from_tensor_slices(my_dataset)" in code
  assert ".shuffle" not in code
  assert ".batch(1)" in code
  assert ".prefetch(tf.data.AUTOTUNE)" in code
