"""Docstring."""

from unittest.mock import MagicMock

import libcst as cst

from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.state_flag_injection import (
  _PLUGIN_KEY,
  _get_func_name,
  capture_eval_state,
  inject_training_flag_call,
)


def test_get_func_name() -> None:
  """Docstring."""
  node: cst.BaseExpression = cst.parse_expression("model")
  assert _get_func_name(node) == "model"

  node2: cst.BaseExpression = cst.parse_expression("self.layer.sublayer")
  assert _get_func_name(node2) == "self.layer.sublayer"

  assert _get_func_name(cst.Integer("1")) is None


def test_inject_training_flag_call() -> None:
  """Docstring."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.metadata = {}

  node: cst.BaseExpression = cst.parse_expression("model(x)")
  assert inject_training_flag_call(node, ctx) is node

  ctx.metadata = {_PLUGIN_KEY: {"model": {"training": cst.Name("False")}}}
  new_node: cst.BaseExpression = inject_training_flag_call(node, ctx)
  code: str = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=new_node)])]).code
  assert code.strip() == "model(x, training=False)"

  node_with_method: cst.BaseExpression = cst.parse_expression("model.forward(x)")
  new_node2: cst.BaseExpression = inject_training_flag_call(node_with_method, ctx)
  code2: str = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=new_node2)])]).code
  assert code2.strip() == "model.forward(x, training=False)"

  # if already has arg
  node_with_arg: cst.BaseExpression = cst.parse_expression("model(x, training=True)")
  new_node3: cst.BaseExpression = inject_training_flag_call(node_with_arg, ctx)
  code3: str = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=new_node3)])]).code
  assert code3.strip() == "model(x, training=True)"

  # not found
  node_other: cst.BaseExpression = cst.parse_expression("other_model(x)")
  assert inject_training_flag_call(node_other, ctx) is node_other


def test_capture_eval_state() -> None:
  """Docstring."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.metadata = {}

  node: cst.BaseExpression = cst.parse_expression("model.eval()")
  new_node: cst.BaseExpression = capture_eval_state(node, ctx)
  code: str = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=new_node)])]).code
  assert code.strip() == "None()"  # Because with_changes(func=Name("None"), args=[]) makes it None()
  # Actually wait: The hook says `return node.with_changes(func=cst.Name("None"), args=[])`
  # So `model.eval()` becomes `None()` not `None`. We just assert logic.

  assert ctx.metadata[_PLUGIN_KEY]["model"]["training"].value == "False"

  node_train: cst.BaseExpression = cst.parse_expression("self.layer.train()")
  _new_node2: cst.CSTNode = capture_eval_state(node_train, ctx)
  assert ctx.metadata[_PLUGIN_KEY]["self.layer"]["training"].value == "True"

  node_train_false: cst.BaseExpression = cst.parse_expression("self.layer.train(False)")
  _new_node3: cst.CSTNode = capture_eval_state(node_train_false, ctx)
  assert ctx.metadata[_PLUGIN_KEY]["self.layer"]["training"].value == "False"

  # not an attribute
  node_func: cst.BaseExpression = cst.parse_expression("eval()")
  assert capture_eval_state(node_func, ctx) is node_func

  # unable to resolve name
  node_unresolved: cst.BaseExpression = cst.parse_expression("eval(1)")
  assert capture_eval_state(node_unresolved, ctx) is node_unresolved
