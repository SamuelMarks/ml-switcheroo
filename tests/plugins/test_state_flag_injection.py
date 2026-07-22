"""Test suite for the State Flag Injection module."""

import libcst as cst
from ml_switcheroo.plugins.state_flag_injection import inject_training_flag_call, capture_eval_state, _get_func_name
from ml_switcheroo.core.hooks import HookContext
from unittest.mock import MagicMock


def test_get_func_name():
  """Gets function name."""
  assert _get_func_name(cst.Name("model")) == "model"
  attr = cst.Attribute(value=cst.Name("self"), attr=cst.Name("layer"))
  assert _get_func_name(attr) == "self.layer"
  assert _get_func_name(cst.Call(func=cst.Name("foo"))) is None


def test_capture_eval_state_not_attribute():
  """Verifies the behavior of capture eval state not attribute."""
  node = cst.Call(func=cst.Name("model"))
  ctx = HookContext(semantics=MagicMock(), config=MagicMock())
  res = capture_eval_state(node, ctx)
  assert res is node


def test_capture_eval_state_eval():
  """Verifies the behavior of capture eval state eval."""
  node = cst.Call(func=cst.Attribute(value=cst.Name("model"), attr=cst.Name("eval")))
  ctx = HookContext(semantics=MagicMock(), config=MagicMock())
  res = capture_eval_state(node, ctx)
  assert res.func.value == "None"
  assert "model" in ctx.metadata["state_flag_injection"]
  assert ctx.metadata["state_flag_injection"]["model"]["training"].value == "False"


def test_capture_eval_state_train():
  """Verifies the behavior of capture eval state train."""
  node = cst.Call(func=cst.Attribute(value=cst.Name("model"), attr=cst.Name("train")))
  ctx = HookContext(semantics=MagicMock(), config=MagicMock())
  res = capture_eval_state(node, ctx)
  assert res.func.value == "None"
  assert "model" in ctx.metadata["state_flag_injection"]
  assert ctx.metadata["state_flag_injection"]["model"]["training"].value == "True"


def test_capture_eval_state_train_with_args():
  """Verifies the behavior of capture eval state train with arguments."""
  node = cst.Call(
    func=cst.Attribute(value=cst.Name("model"), attr=cst.Name("train")), args=[cst.Arg(value=cst.Name("False"))]
  )
  ctx = HookContext(semantics=MagicMock(), config=MagicMock())
  _res = capture_eval_state(node, ctx)
  assert ctx.metadata["state_flag_injection"]["model"]["training"].value == "False"


def test_inject_training_flag_call_no_store():
  """Injects training flag call no store."""
  node = cst.Call(func=cst.Name("model"))
  ctx = HookContext(semantics=MagicMock(), config=MagicMock())
  res = inject_training_flag_call(node, ctx)
  assert res is node


def test_inject_training_flag_call_no_match():
  """Injects training flag call no match."""
  node = cst.Call(func=cst.Name("model"))
  ctx = HookContext(semantics=MagicMock(), config=MagicMock())
  ctx.metadata["state_flag_injection"] = {"other_model": {"training": cst.Name("False")}}
  res = inject_training_flag_call(node, ctx)
  assert res is node


def test_inject_training_flag_call_implicit():
  """Injects training flag call implicit."""
  node = cst.Call(func=cst.Name("model"))
  ctx = HookContext(semantics=MagicMock(), config=MagicMock())
  ctx.metadata["state_flag_injection"] = {"model": {"training": cst.Name("False")}}
  res = inject_training_flag_call(node, ctx)
  assert len(res.args) == 1
  assert res.args[0].keyword.value == "training"
  assert res.args[0].value.value == "False"


def test_inject_training_flag_call_explicit():
  """Injects training flag call explicit."""
  node = cst.Call(
    func=cst.Attribute(value=cst.Name("model"), attr=cst.Name("forward")),
    args=[cst.Arg(value=cst.Name("x"), comma=cst.Comma())],
  )
  ctx = HookContext(semantics=MagicMock(), config=MagicMock())
  ctx.metadata["state_flag_injection"] = {"model": {"training": cst.Name("True")}}
  res = inject_training_flag_call(node, ctx)
  assert len(res.args) == 2
  assert res.args[1].keyword.value == "training"
  assert res.args[1].value.value == "True"


def test_get_func_name_base_none():
  """Gets function name base none."""
  attr = cst.Attribute(value=cst.Call(func=cst.Name("foo")), attr=cst.Name("bar"))
  assert _get_func_name(attr) is None


def test_inject_training_flag_call_func_none():
  """Injects training flag call function none."""
  node = cst.Call(func=cst.Call(func=cst.Name("foo")))
  ctx = HookContext(semantics=MagicMock(), config=MagicMock())
  ctx.metadata["state_flag_injection"] = {"foo": {"training": cst.Name("True")}}
  res = inject_training_flag_call(node, ctx)
  assert res is node


def test_inject_training_flag_call_parent_none():
  """Injects training flag call parent none."""
  node = cst.Call(func=cst.Attribute(value=cst.Call(func=cst.Name("foo")), attr=cst.Name("bar")))
  ctx = HookContext(semantics=MagicMock(), config=MagicMock())
  ctx.metadata["state_flag_injection"] = {"foo": {"training": cst.Name("True")}}
  res = inject_training_flag_call(node, ctx)
  assert res is node


def test_inject_training_flag_call_default_comma():
  """Injects training flag call default comma."""
  node = cst.Call(func=cst.Name("model"), args=[cst.Arg(value=cst.Name("x"))])
  ctx = HookContext(semantics=MagicMock(), config=MagicMock())
  ctx.metadata["state_flag_injection"] = {"model": {"training": cst.Name("True")}}
  res = inject_training_flag_call(node, ctx)
  assert len(res.args) == 2


def test_capture_eval_state_unknown_method():
  """Verifies the behavior of capture eval state unknown method."""
  node = cst.Call(func=cst.Attribute(value=cst.Name("model"), attr=cst.Name("unknown")))
  ctx = HookContext(semantics=MagicMock(), config=MagicMock())
  res = capture_eval_state(node, ctx)
  assert res.func.value == "None"


def test_capture_eval_state_already_in_store():
  """Verifies the behavior of capture eval state already in store."""
  node = cst.Call(func=cst.Attribute(value=cst.Name("model"), attr=cst.Name("train")))
  ctx = HookContext(semantics=MagicMock(), config=MagicMock())
  ctx.metadata["state_flag_injection"] = {"model": {}}
  capture_eval_state(node, ctx)
  assert ctx.metadata["state_flag_injection"]["model"]["training"].value == "True"
