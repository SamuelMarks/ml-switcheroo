"""Test suite for the Plugin Loss Wrapper module."""

from unittest.mock import MagicMock
import libcst as cst
from ml_switcheroo.plugins.loss_wrapper import transform_loss_reduction


def get_dummy_ctx(target_fw: str = "torch", current_op_id: str = "CrossEntropyLoss") -> MagicMock:
  """Docstring."""
  ctx = MagicMock()
  ctx.target_fw = target_fw
  ctx.current_op_id = current_op_id
  ctx.semantics.get_operation.return_value = MagicMock(is_loss=True)
  return ctx


def test_plugin_loss_wrapper_mean() -> None:
  """Docstring."""
  ctx = get_dummy_ctx()
  ctx.lookup_api.side_effect = lambda x: "dummy.loss" if x == "CrossEntropyLoss" else "dummy.mean"
  # no reduction arg -> mean by default
  node = cst.Call(func=cst.Name("dummy"), args=[cst.Arg(value=cst.Name("input"))])
  res = transform_loss_reduction(node, ctx)
  assert isinstance(res, cst.Call)


def test_plugin_loss_wrapper_sum() -> None:
  """Docstring."""
  ctx = get_dummy_ctx()
  ctx.lookup_api.side_effect = lambda x: "dummy.loss" if x == "CrossEntropyLoss" else "dummy.sum"
  node = cst.Call(
    func=cst.Name("dummy"),
    args=[cst.Arg(value=cst.Name("input")), cst.Arg(value=cst.SimpleString('"sum"'), keyword=cst.Name("reduction"))],
  )
  res = transform_loss_reduction(node, ctx)
  assert isinstance(res, cst.Call)


def test_plugin_loss_wrapper_none() -> None:
  """Docstring."""
  ctx = get_dummy_ctx()
  ctx.lookup_api.side_effect = lambda x: "dummy.loss" if x == "CrossEntropyLoss" else None
  node = cst.Call(func=cst.Name("dummy"), args=[cst.Arg(value=cst.SimpleString('"none"'), keyword=cst.Name("reduction"))])
  res = transform_loss_reduction(node, ctx)
  assert isinstance(res, cst.Call)


def test_plugin_loss_wrapper_other_keyword() -> None:
  """Docstring."""
  ctx = get_dummy_ctx()
  ctx.lookup_api.side_effect = lambda x: "dummy.loss" if x == "CrossEntropyLoss" else "dummy.mean"
  node = cst.Call(
    func=cst.Name("dummy"),
    args=[
      cst.Arg(value=cst.Name("foo"), keyword=cst.Name("other")),
      cst.Arg(value=cst.SimpleString('"sum"'), keyword=cst.Name("reduction")),
    ],
  )
  res = transform_loss_reduction(node, ctx)
  assert isinstance(res, cst.Call)


def test_plugin_loss_wrapper_reduction_non_string() -> None:
  """Docstring."""
  ctx = get_dummy_ctx()
  ctx.lookup_api.side_effect = lambda x: "dummy.loss" if x == "CrossEntropyLoss" else "dummy.mean"
  node = cst.Call(func=cst.Name("dummy"), args=[cst.Arg(value=cst.Integer("1"), keyword=cst.Name("reduction"))])
  res = transform_loss_reduction(node, ctx)
  assert isinstance(res, cst.Call)


def test_plugin_loss_wrapper_reduction_name() -> None:
  """Docstring."""
  ctx = get_dummy_ctx()
  ctx.lookup_api.side_effect = lambda x: "dummy.loss" if x == "CrossEntropyLoss" else "dummy.mean"
  node = cst.Call(func=cst.Name("dummy"), args=[cst.Arg(value=cst.Name("my_mode"), keyword=cst.Name("reduction"))])
  res = transform_loss_reduction(node, ctx)
  assert isinstance(res, cst.Call)


def test_plugin_loss_wrapper_none_ctx() -> None:
  """Docstring."""
  ctx = get_dummy_ctx(current_op_id=None)
  ctx.lookup_api.side_effect = lambda x: "dummy.loss" if x == "CrossEntropyLoss" else "dummy.mean"
  node = cst.Call(func=cst.Name("dummy"), args=[])
  res = transform_loss_reduction(node, ctx)
  assert isinstance(res, cst.Call)


def test_plugin_loss_wrapper_no_api() -> None:
  """Docstring."""
  ctx = get_dummy_ctx()
  ctx.lookup_api.return_value = None
  node = cst.Call(func=cst.Name("dummy"), args=[])
  res = transform_loss_reduction(node, ctx)
  # returns original node
  assert res is node


def test_plugin_loss_wrapper_no_wrapper_api() -> None:
  """Docstring."""
  ctx = get_dummy_ctx()
  ctx.lookup_api.side_effect = lambda x: "dummy.loss" if x == "CrossEntropyLoss" else None
  node = cst.Call(func=cst.Name("dummy"), args=[])
  res = transform_loss_reduction(node, ctx)
  assert isinstance(res, cst.Call)
