"""Test suite for the State Flag Injection module."""

from unittest.mock import MagicMock

import libcst as cst

from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.state_flag_injection import _get_func_name, capture_eval_state, inject_training_flag_call


def test_get_func_name() -> None:
  """Gets function name."""
  assert _get_func_name(cst.Name("model")) == "model"
  attr: cst.Attribute = cst.Attribute(value=cst.Name("self"), attr=cst.Name("layer"))
  assert _get_func_name(attr) == "self.layer"
  assert _get_func_name(cst.Call(func=cst.Name("foo"))) is None


def test_capture_eval_state_not_attribute() -> None:
  """Verify `Call` handles capture eval state not attribute as expected."""
  node: cst.Call = cst.Call(func=cst.Name("model"))
  ctx: HookContext = HookContext(semantics=MagicMock(), config=MagicMock())
  res: cst.CSTNode | cst.SimpleString | cst.Name | cst.Call = capture_eval_state(node, ctx)
  assert res is node


def test_capture_eval_state_eval() -> None:
  """Verify `Call` handles capture eval state eval as expected."""
  node: cst.Call = cst.Call(func=cst.Attribute(value=cst.Name("model"), attr=cst.Name("eval")))
  ctx: HookContext = HookContext(semantics=MagicMock(), config=MagicMock())
  res: cst.CSTNode | cst.SimpleString | cst.Name | cst.Call = capture_eval_state(node, ctx)
  assert isinstance(res, cst.Call)
  assert isinstance(res.func, cst.Name)
  assert res.func.value == "None"
  store = ctx.metadata["state_flag_injection"]
  assert isinstance(store, dict)
  assert "model" in store
  model_state = store["model"]
  assert isinstance(model_state, dict)
  training_expr = model_state["training"]
  assert isinstance(training_expr, cst.Name)
  assert training_expr.value == "False"


def test_capture_eval_state_train() -> None:
  """Verify `Call` handles capture eval state train as expected."""
  node: cst.Call = cst.Call(func=cst.Attribute(value=cst.Name("model"), attr=cst.Name("train")))
  ctx: HookContext = HookContext(semantics=MagicMock(), config=MagicMock())
  res: cst.CSTNode | cst.SimpleString | cst.Name | cst.Call = capture_eval_state(node, ctx)
  assert isinstance(res, cst.Call)
  assert isinstance(res.func, cst.Name)
  assert res.func.value == "None"
  store = ctx.metadata["state_flag_injection"]
  assert isinstance(store, dict)
  assert "model" in store
  model_state = store["model"]
  assert isinstance(model_state, dict)
  training_expr = model_state["training"]
  assert isinstance(training_expr, cst.Name)
  assert training_expr.value == "True"


def test_capture_eval_state_train_with_args() -> None:
  """Verify `Call` handles capture eval state train with args as expected."""
  node: cst.Call = cst.Call(
    func=cst.Attribute(value=cst.Name("model"), attr=cst.Name("train")), args=[cst.Arg(value=cst.Name("False"))]
  )
  ctx: HookContext = HookContext(semantics=MagicMock(), config=MagicMock())
  _res: cst.CSTNode | cst.SimpleString | cst.Name | cst.Call = capture_eval_state(node, ctx)
  store = ctx.metadata["state_flag_injection"]
  assert isinstance(store, dict)
  model_state = store["model"]
  assert isinstance(model_state, dict)
  training_expr = model_state["training"]
  assert isinstance(training_expr, cst.Name)
  assert training_expr.value == "False"


def test_inject_training_flag_call_no_store() -> None:
  """Injects training flag call no store."""
  node: cst.Call = cst.Call(func=cst.Name("model"))
  ctx: HookContext = HookContext(semantics=MagicMock(), config=MagicMock())
  res: cst.CSTNode | cst.Call = inject_training_flag_call(node, ctx)
  assert res is node


def test_inject_training_flag_call_no_match() -> None:
  """Injects training flag call no match."""
  node: cst.Call = cst.Call(func=cst.Name("model"))
  ctx: HookContext = HookContext(semantics=MagicMock(), config=MagicMock())
  store: dict[str, dict[str, cst.BaseExpression]] = {
    "other_model": {"training": cst.Name("False")},
  }
  ctx.metadata["state_flag_injection"] = store
  res: cst.CSTNode | cst.Call = inject_training_flag_call(node, ctx)
  assert res is node


def test_inject_training_flag_call_implicit() -> None:
  """Injects training flag call implicit."""
  node: cst.Call = cst.Call(func=cst.Name("model"))
  ctx: HookContext = HookContext(semantics=MagicMock(), config=MagicMock())
  store: dict[str, dict[str, cst.BaseExpression]] = {
    "model": {"training": cst.Name("False")},
  }
  ctx.metadata["state_flag_injection"] = store
  res: cst.CSTNode | cst.Call = inject_training_flag_call(node, ctx)
  assert isinstance(res, cst.Call)
  assert len(res.args) == 1
  assert res.args[0].keyword is not None
  assert res.args[0].keyword.value == "training"
  assert isinstance(res.args[0].value, cst.Name)
  assert res.args[0].value.value == "False"


def test_inject_training_flag_call_explicit() -> None:
  """Injects training flag call explicit."""
  node: cst.Call = cst.Call(
    func=cst.Attribute(value=cst.Name("model"), attr=cst.Name("forward")),
    args=[cst.Arg(value=cst.Name("x"), comma=cst.Comma())],
  )
  ctx: HookContext = HookContext(semantics=MagicMock(), config=MagicMock())
  store: dict[str, dict[str, cst.BaseExpression]] = {
    "model": {"training": cst.Name("True")},
  }
  ctx.metadata["state_flag_injection"] = store
  res: cst.CSTNode | cst.Call = inject_training_flag_call(node, ctx)
  assert isinstance(res, cst.Call)
  assert len(res.args) == 2
  assert res.args[1].keyword is not None
  assert res.args[1].keyword.value == "training"
  assert isinstance(res.args[1].value, cst.Name)
  assert res.args[1].value.value == "True"


def test_get_func_name_base_none() -> None:
  """Gets function name base none."""
  attr: cst.Attribute = cst.Attribute(value=cst.Call(func=cst.Name("foo")), attr=cst.Name("bar"))
  assert _get_func_name(attr) is None


def test_inject_training_flag_call_func_none() -> None:
  """Injects training flag call function none."""
  node: cst.Call = cst.Call(func=cst.Call(func=cst.Name("foo")))
  ctx: HookContext = HookContext(semantics=MagicMock(), config=MagicMock())
  store: dict[str, dict[str, cst.BaseExpression]] = {
    "foo": {"training": cst.Name("True")},
  }
  ctx.metadata["state_flag_injection"] = store
  res: cst.CSTNode | cst.Call = inject_training_flag_call(node, ctx)
  assert res is node


def test_inject_training_flag_call_parent_none() -> None:
  """Injects training flag call parent none."""
  node: cst.Call = cst.Call(func=cst.Attribute(value=cst.Call(func=cst.Name("foo")), attr=cst.Name("bar")))
  ctx: HookContext = HookContext(semantics=MagicMock(), config=MagicMock())
  store: dict[str, dict[str, cst.BaseExpression]] = {
    "foo": {"training": cst.Name("True")},
  }
  ctx.metadata["state_flag_injection"] = store
  res: cst.CSTNode | cst.Call = inject_training_flag_call(node, ctx)
  assert res is node


def test_inject_training_flag_call_default_comma() -> None:
  """Injects training flag call default comma."""
  node: cst.Call = cst.Call(func=cst.Name("model"), args=[cst.Arg(value=cst.Name("x"))])
  ctx: HookContext = HookContext(semantics=MagicMock(), config=MagicMock())
  store: dict[str, dict[str, cst.BaseExpression]] = {
    "model": {"training": cst.Name("True")},
  }
  ctx.metadata["state_flag_injection"] = store
  res: cst.CSTNode | cst.Call = inject_training_flag_call(node, ctx)
  assert isinstance(res, cst.Call)
  assert len(res.args) == 2


def test_inject_training_flag_call_duplicate() -> None:
  """Verifies that flag is not injected if already present."""
  node: cst.Call = cst.Call(
    func=cst.Name("model"),
    args=[cst.Arg(value=cst.Name("x")), cst.Arg(keyword=cst.Name("training"), value=cst.Name("False"))],
  )
  ctx: HookContext = HookContext(semantics=MagicMock(), config=MagicMock())
  store: dict[str, dict[str, cst.BaseExpression]] = {
    "model": {"training": cst.Name("True")},
  }
  ctx.metadata["state_flag_injection"] = store
  res: cst.CSTNode | cst.Call = inject_training_flag_call(node, ctx)
  assert isinstance(res, cst.Call)
  assert len(res.args) == 2
  assert isinstance(res.args[1].value, cst.Name)
  assert res.args[1].value.value == "False"


def test_capture_eval_state_func_name_none() -> None:
  """Verifies behavior when _get_func_name returns None during capture."""
  node: cst.Call = cst.Call(func=cst.Attribute(value=cst.Call(func=cst.Name("foo")), attr=cst.Name("eval")))
  ctx: HookContext = HookContext(semantics=MagicMock(), config=MagicMock())
  res: cst.CSTNode | cst.SimpleString | cst.Name | cst.Call = capture_eval_state(node, ctx)
  assert res is node


def test_capture_eval_state_unknown_method() -> None:
  """Verify `Call` handles capture eval state unknown method as expected."""
  node: cst.Call = cst.Call(func=cst.Attribute(value=cst.Name("model"), attr=cst.Name("unknown")))
  ctx: HookContext = HookContext(semantics=MagicMock(), config=MagicMock())
  res: cst.CSTNode | cst.SimpleString | cst.Name | cst.Call = capture_eval_state(node, ctx)
  assert isinstance(res, cst.Call)
  assert isinstance(res.func, cst.Name)
  assert res.func.value == "None"


def test_capture_eval_state_already_in_store() -> None:
  """Verify `Call` handles capture eval state already in store as expected."""
  node: cst.Call = cst.Call(func=cst.Attribute(value=cst.Name("model"), attr=cst.Name("train")))
  ctx: HookContext = HookContext(semantics=MagicMock(), config=MagicMock())
  store: dict[str, dict[str, cst.BaseExpression]] = {"model": {}}
  ctx.metadata["state_flag_injection"] = store
  capture_eval_state(node, ctx)
  training_expr = store["model"]["training"]
  assert isinstance(training_expr, cst.Name)
  assert training_expr.value == "True"
