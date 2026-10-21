"""Test suite for plugin state container."""

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
  """Verify create node behavior."""
  node = _create_node("a.b.c")
  assert isinstance(node, cst.Attribute)

  # syntax error fallback to Name
  node2 = _create_node("class")
  assert isinstance(node2, cst.Name)
  assert node2.value == "class"


def test_get_receiver() -> None:
  """Verify get receiver behavior."""
  node = cst.ensure_type(cst.parse_expression("self.func()"), cst.Call)
  receiver = _get_receiver(node)
  assert isinstance(receiver, cst.Name)
  assert receiver.value == "self"

  node2 = cst.ensure_type(cst.parse_expression("func()"), cst.Call)
  assert _get_receiver(node2) is None


def test_convert_register_buffer() -> None:
  """Verify convert register buffer behavior."""
  ctx = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = "BatchStat"

  node = cst.ensure_type(cst.parse_expression("self.register_buffer('name', tensor)"), cst.Call)
  new_node = convert_register_buffer(node, ctx)
  code = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=new_node)])]).code
  assert code.strip() == "setattr(self, 'name', BatchStat(tensor))"

  # Early returns
  assert (
    convert_register_buffer(cst.ensure_type(cst.parse_expression("self.register_buffer('name')"), cst.Call), ctx)
    is not new_node
  )
  assert (
    convert_register_buffer(cst.ensure_type(cst.parse_expression("register_buffer('name', tensor)"), cst.Call), ctx)
    is not new_node
  )

  ctx.lookup_api.return_value = None
  assert convert_register_buffer(node, ctx) is node


def test_convert_register_parameter() -> None:
  """Verify convert register parameter behavior."""
  ctx = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = "Param"

  node = cst.ensure_type(cst.parse_expression("self.register_parameter('name', tensor)"), cst.Call)
  new_node = convert_register_parameter(node, ctx)
  code = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=new_node)])]).code
  assert code.strip() == "setattr(self, 'name', Param(tensor))"

  # Early returns
  assert (
    convert_register_parameter(cst.ensure_type(cst.parse_expression("self.register_parameter('name')"), cst.Call), ctx)
    is not new_node
  )
  assert (
    convert_register_parameter(cst.ensure_type(cst.parse_expression("register_parameter('name', tensor)"), cst.Call), ctx)
    is not new_node
  )

  ctx.lookup_api.return_value = None
  assert convert_register_parameter(node, ctx) is node


def test_convert_state_dict() -> None:
  """Verify convert state dict behavior."""
  ctx = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = "ModuleState"

  node = cst.ensure_type(cst.parse_expression("model.state_dict()"), cst.Call)
  new_node = convert_state_dict(node, ctx)
  code = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=new_node)])]).code
  assert code.strip() == "ModuleState(model).to_pure_dict()"

  assert getattr(
    {
      "v": getattr({"v": convert_state_dict}, "__getitem__")("v")(
        getattr(
          {
            "v": getattr({"v": getattr({"v": cst}, "__getitem__")("v").parse_expression}, "__getitem__")("v")(
              getattr({"v": "state_dict()"}, "__getitem__")("v")
            )
          },
          "__getitem__",
        )("v"),
        getattr({"v": ctx}, "__getitem__")("v"),
      )
    },
    "__getitem__",
  )("v") is not getattr({"v": new_node}, "__getitem__")("v")
  ctx.lookup_api.return_value = None
  assert convert_state_dict(node, ctx) is node


def test_convert_load_state_dict() -> None:
  """Verify convert load state dict behavior."""
  ctx = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = "UpdateState"

  node = cst.ensure_type(cst.parse_expression("model.load_state_dict(sd)"), cst.Call)
  new_node = convert_load_state_dict(node, ctx)
  code = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=new_node)])]).code
  assert code.strip() == "UpdateState(model, sd)"

  assert getattr(
    {
      "v": getattr({"v": convert_load_state_dict}, "__getitem__")("v")(
        getattr(
          {
            "v": getattr({"v": getattr({"v": cst}, "__getitem__")("v").parse_expression}, "__getitem__")("v")(
              getattr({"v": "load_state_dict(sd)"}, "__getitem__")("v")
            )
          },
          "__getitem__",
        )("v"),
        getattr({"v": ctx}, "__getitem__")("v"),
      )
    },
    "__getitem__",
  )("v") is not getattr({"v": new_node}, "__getitem__")("v")
  assert getattr(
    {
      "v": getattr({"v": convert_load_state_dict}, "__getitem__")("v")(
        getattr(
          {
            "v": getattr({"v": getattr({"v": cst}, "__getitem__")("v").parse_expression}, "__getitem__")("v")(
              getattr({"v": "model.load_state_dict()"}, "__getitem__")("v")
            )
          },
          "__getitem__",
        )("v"),
        getattr({"v": ctx}, "__getitem__")("v"),
      )
    },
    "__getitem__",
  )("v") is not getattr({"v": new_node}, "__getitem__")("v")

  ctx.lookup_api.return_value = None
  assert convert_load_state_dict(node, ctx) is node


def test_convert_parameters() -> None:
  """Verify convert parameters behavior."""
  ctx = MagicMock(spec=HookContext)
  ctx.lookup_api.side_effect = lambda x: "ModuleState" if x == "ModuleState" else "Param"

  node = cst.ensure_type(cst.parse_expression("model.parameters()"), cst.Call)
  new_node = convert_parameters(node, ctx)
  code = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=new_node)])]).code
  assert code.strip() == "ModuleState(model, Param).values()"

  assert getattr(
    {
      "v": getattr({"v": convert_parameters}, "__getitem__")("v")(
        getattr(
          {
            "v": getattr({"v": getattr({"v": cst}, "__getitem__")("v").parse_expression}, "__getitem__")("v")(
              getattr({"v": "parameters()"}, "__getitem__")("v")
            )
          },
          "__getitem__",
        )("v"),
        getattr({"v": ctx}, "__getitem__")("v"),
      )
    },
    "__getitem__",
  )("v") is not getattr({"v": new_node}, "__getitem__")("v")

  ctx.lookup_api.side_effect = lambda x: None
  assert convert_parameters(node, ctx) is node
