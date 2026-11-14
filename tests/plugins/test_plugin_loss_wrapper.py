"""Test suite for the Plugin Loss Wrapper module."""

from unittest.mock import MagicMock

import libcst as cst

from ml_switcheroo.plugins.loss_wrapper import transform_loss_reduction


def get_dummy_ctx(target_framework: str = "torch", current_op_id: str | None = "CrossEntropyLoss") -> MagicMock:
  """Execute `get_dummy_ctx` helper in plugin loss wrapper tests.

  Args:
      target_framework: Target framework identifier string.
      current_op_id: Input `current_op_id` parameter value (`str | None`).

  Returns:
      Resulting `MagicMock` from get dummy ctx.
  """
  ctx = MagicMock()
  ctx.target_framework = target_framework
  ctx.current_op_id = current_op_id
  ctx.semantics.get_operation.return_value = MagicMock(is_loss=True)
  return ctx


def test_plugin_loss_wrapper_mean() -> None:
  """Verify `get_dummy_ctx` handles plugin loss wrapper mean as expected."""
  ctx = get_dummy_ctx()
  ctx.lookup_api.side_effect = lambda x: "dummy.loss" if x == "CrossEntropyLoss" else "dummy.mean"
  # no reduction arg -> mean by default
  node = cst.Call(func=cst.Name("dummy"), args=[cst.Arg(value=cst.Name("input"))])
  res = transform_loss_reduction(node, ctx)
  assert isinstance(res, cst.Call)
  assert cst.Module([]).code_for_node(res) == "dummy.mean(dummy.loss(input))"


def test_plugin_loss_wrapper_sum() -> None:
  """Verify `get_dummy_ctx` handles plugin loss wrapper sum as expected."""
  ctx = get_dummy_ctx()
  ctx.lookup_api.side_effect = lambda x: "dummy.loss" if x == "CrossEntropyLoss" else "dummy.sum"
  node = cst.Call(
    func=cst.Name("dummy"),
    args=[cst.Arg(value=cst.Name("input")), cst.Arg(value=cst.SimpleString('"sum"'), keyword=cst.Name("reduction"))],
  )
  res = transform_loss_reduction(node, ctx)
  assert isinstance(res, cst.Call)
  assert cst.Module([]).code_for_node(res) == "dummy.sum(dummy.loss(input))"


def test_plugin_loss_wrapper_none() -> None:
  """Verify `get_dummy_ctx` handles plugin loss wrapper none as expected."""
  ctx = get_dummy_ctx()
  ctx.lookup_api.side_effect = lambda x: "dummy.loss" if x == "CrossEntropyLoss" else None
  node = cst.Call(func=cst.Name("dummy"), args=[cst.Arg(value=cst.SimpleString('"none"'), keyword=cst.Name("reduction"))])
  res = transform_loss_reduction(node, ctx)
  assert isinstance(res, cst.Call)
  assert len(res.args) == 0
  assert cst.Module([]).code_for_node(res) == "dummy.loss()"


def test_plugin_loss_wrapper_other_keyword() -> None:
  """Verify `get_dummy_ctx` handles plugin loss wrapper other keyword as expected."""
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
  assert len(res.args) == 1
  inner = res.args[0].value
  assert isinstance(inner, cst.Call)
  assert len(inner.args) == 1
  assert cst.ensure_type(inner.args[0].keyword, cst.Name).value == "other"
  assert cst.ensure_type(inner.args[0].value, cst.Name).value == "foo"


def test_plugin_loss_wrapper_reduction_non_string() -> None:
  """Verify `get_dummy_ctx` handles plugin loss wrapper reduction non string as expected."""
  ctx = get_dummy_ctx()
  ctx.lookup_api.side_effect = lambda x: "dummy.loss" if x == "CrossEntropyLoss" else "dummy.mean"
  node = cst.Call(func=cst.Name("dummy"), args=[cst.Arg(value=cst.Integer("1"), keyword=cst.Name("reduction"))])
  res = transform_loss_reduction(node, ctx)
  assert isinstance(res, cst.Call)
  assert cst.Module([]).code_for_node(res) == "dummy.mean(dummy.loss())"


def test_plugin_loss_wrapper_reduction_name() -> None:
  """Verify `get_dummy_ctx` handles plugin loss wrapper reduction name as expected."""
  ctx = get_dummy_ctx()
  ctx.lookup_api.side_effect = lambda x: "dummy.loss" if x == "CrossEntropyLoss" else "dummy.mean"
  node = cst.Call(func=cst.Name("dummy"), args=[cst.Arg(value=cst.Name("my_mode"), keyword=cst.Name("reduction"))])
  res = transform_loss_reduction(node, ctx)
  assert isinstance(res, cst.Call)
  assert cst.Module([]).code_for_node(res) == "dummy.mean(dummy.loss())"


def test_plugin_loss_wrapper_none_ctx() -> None:
  """Verify `get_dummy_ctx` handles plugin loss wrapper none ctx as expected."""
  ctx = get_dummy_ctx(current_op_id=None)
  ctx.lookup_api.side_effect = lambda x: "dummy.loss" if x == "CrossEntropyLoss" else "dummy.mean"
  node = cst.Call(func=cst.Name("dummy"), args=[])
  res = transform_loss_reduction(node, ctx)
  assert isinstance(res, cst.Call)
  assert cst.Module([]).code_for_node(res) == "dummy.mean(dummy.loss())"


def test_plugin_loss_wrapper_no_api() -> None:
  """Verify `get_dummy_ctx` handles plugin loss wrapper no API as expected."""
  ctx = get_dummy_ctx()
  ctx.lookup_api.return_value = None
  node = cst.Call(func=cst.Name("dummy"), args=[])
  res = transform_loss_reduction(node, ctx)
  # returns original node
  assert res is node


def test_plugin_loss_wrapper_no_wrapper_api() -> None:
  """Verify `get_dummy_ctx` handles plugin loss wrapper no wrapper API as expected."""
  ctx = get_dummy_ctx()
  ctx.lookup_api.side_effect = lambda x: "dummy.loss" if x == "CrossEntropyLoss" else None
  node = cst.Call(func=cst.Name("dummy"), args=[])
  res = transform_loss_reduction(node, ctx)
  assert isinstance(res, cst.Call)
  assert len(res.args) == 0
  assert cst.Module([]).code_for_node(res) == "dummy.loss()"
