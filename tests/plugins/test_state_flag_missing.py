import pytest
import libcst as cst
from unittest.mock import MagicMock
from ml_switcheroo.plugins.state_flag_injection import _get_func_name, inject_training_flag_call, capture_eval_state
from ml_switcheroo.core.hooks import HookContext


def test_get_func_name():
  assert _get_func_name(cst.Call(func=cst.SimpleString("'a'"))) is None


def test_inject_training_flag_call():
  ctx = MagicMock(spec=HookContext)
  ctx.metadata = {}

  # 96: None target_api
  ctx.lookup_api.return_value = None
  node1 = cst.Call(func=cst.Name("dropout"))
  assert inject_training_flag_call(node1, ctx) == node1

  ctx.lookup_api.return_value = "Dropout"
  # 104: arg has no keyword
  node_pos = cst.Call(func=cst.Name("dropout"), args=[cst.Arg(value=cst.Name("x")), cst.Arg(value=cst.Name("training"))])
  res = inject_training_flag_call(node_pos, ctx)


def test_capture_eval_state():
  ctx = MagicMock(spec=HookContext)
  ctx.metadata = {}

  # 143: not eval or train
  node1 = cst.Call(func=cst.Name("foo"))
  assert capture_eval_state(node1, ctx) == node1

  # 150: method of self or missing?
  node_not_method = cst.Call(func=cst.Name("eval"))
  assert capture_eval_state(node_not_method, ctx) == node_not_method

  # 165: train with positional arg
  node_train_pos = cst.Call(
    func=cst.Attribute(value=cst.Name("model"), attr=cst.Name("train")), args=[cst.Arg(value=cst.Name("False"))]
  )
  res = capture_eval_state(node_train_pos, ctx)
