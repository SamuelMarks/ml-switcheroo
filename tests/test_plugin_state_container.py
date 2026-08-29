"""Docstring."""

from typing import Optional
from unittest.mock import MagicMock

import libcst as cst

from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.state_container import (
  _create_node,
  _get_receiver,
  convert_load_state_dict,
  convert_parameters,
  convert_register_buffer,
  convert_register_parameter,
  convert_state_dict,
)


def test_create_node() -> None:
  """Docstring."""
  node: cst.BaseExpression = _create_node("a.b.c")
  assert isinstance(node, cst.Attribute)

  # syntax error fallback to Name
  node2: cst.BaseExpression = _create_node("class")
  assert isinstance(node2, cst.Name)
  assert node2.value == "class"


def test_get_receiver() -> None:
  """Docstring."""
  node: cst.BaseExpression = cst.parse_expression("self.func()")
  receiver: Optional[cst.BaseExpression] = _get_receiver(node)
  assert isinstance(receiver, cst.Name)
  assert receiver.value == "self"

  node2: cst.BaseExpression = cst.parse_expression("func()")
  assert _get_receiver(node2) is None


def test_convert_register_buffer() -> None:
  """Docstring."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = "BatchStat"

  node: cst.BaseExpression = cst.parse_expression("self.register_buffer('name', tensor)")
  new_node: cst.CSTNode = convert_register_buffer(node, ctx)
  code: str = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=new_node)])]).code
  assert code.strip() == "setattr(self, 'name', BatchStat(tensor))"

  # Early returns
  assert convert_register_buffer(cst.parse_expression("self.register_buffer('name')"), ctx) is not new_node
  assert convert_register_buffer(cst.parse_expression("register_buffer('name', tensor)"), ctx) is not new_node

  ctx.lookup_api.return_value = None
  assert convert_register_buffer(node, ctx) is node


def test_convert_register_parameter() -> None:
  """Docstring."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = "Param"

  node: cst.BaseExpression = cst.parse_expression("self.register_parameter('name', tensor)")
  new_node: cst.CSTNode = convert_register_parameter(node, ctx)
  code: str = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=new_node)])]).code
  assert code.strip() == "setattr(self, 'name', Param(tensor))"

  # Early returns
  assert convert_register_parameter(cst.parse_expression("self.register_parameter('name')"), ctx) is not new_node
  assert convert_register_parameter(cst.parse_expression("register_parameter('name', tensor)"), ctx) is not new_node

  ctx.lookup_api.return_value = None
  assert convert_register_parameter(node, ctx) is node


def test_convert_state_dict() -> None:
  """Docstring."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = "ModuleState"

  node: cst.BaseExpression = cst.parse_expression("model.state_dict()")
  new_node: cst.CSTNode = convert_state_dict(node, ctx)
  code: str = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=new_node)])]).code
  assert code.strip() == "ModuleState(model).to_pure_dict()"

  assert convert_state_dict(cst.parse_expression("state_dict()"), ctx) is not new_node
  ctx.lookup_api.return_value = None
  assert convert_state_dict(node, ctx) is node


def test_convert_load_state_dict() -> None:
  """Docstring."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = "UpdateState"

  node: cst.BaseExpression = cst.parse_expression("model.load_state_dict(sd)")
  new_node: cst.CSTNode = convert_load_state_dict(node, ctx)
  code: str = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=new_node)])]).code
  assert code.strip() == "UpdateState(model, sd)"

  assert convert_load_state_dict(cst.parse_expression("load_state_dict(sd)"), ctx) is not new_node
  assert convert_load_state_dict(cst.parse_expression("model.load_state_dict()"), ctx) is not new_node

  ctx.lookup_api.return_value = None
  assert convert_load_state_dict(node, ctx) is node


def test_convert_parameters() -> None:
  """Docstring."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.lookup_api.side_effect = lambda x: "ModuleState" if x == "ModuleState" else "Param"

  node: cst.BaseExpression = cst.parse_expression("model.parameters()")
  new_node: cst.CSTNode = convert_parameters(node, ctx)
  code: str = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=new_node)])]).code
  assert code.strip() == "ModuleState(model, Param).values()"

  assert convert_parameters(cst.parse_expression("parameters()"), ctx) is not new_node

  ctx.lookup_api.side_effect = lambda x: None
  assert convert_parameters(node, ctx) is node
