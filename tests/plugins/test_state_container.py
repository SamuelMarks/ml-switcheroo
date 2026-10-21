"""Test suite for the State Container module."""

from unittest.mock import patch

import libcst as cst

from ml_switcheroo.plugins.state_container import (
  _create_node,
  _get_receiver,
  convert_load_state_dict,
  convert_parameters,
  convert_register_buffer,
  convert_register_parameter,
  convert_state_dict,
)


class DummyContext:
  """Test helper class DummyContext."""

  def __init__(self, api_map: dict[str, str] | None = None) -> None:
    """Initializes the DummyContext instance.

    Args:
        api_map: Test argument or fixture.
    """
    self.api_map: dict[str, str] = api_map or {}

  def lookup_api(self, op_id: str) -> str | None:
    """Mock implementation of lookup API.

    Args:
        op_id: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self.api_map.get(op_id)


def test_create_node() -> None:
  """Creates node."""
  node = _create_node("a.b.c")
  assert isinstance(node, cst.Attribute)
  with patch("libcst.parse_expression", side_effect=Exception("Failed")):
    node = _create_node("fallback_name")
  assert isinstance(node, cst.Name)
  assert node.value == "fallback_name"


def test_get_receiver() -> None:
  """Gets receiver."""
  node = cst.ensure_type(cst.parse_expression("self.register_buffer('name', tensor)"), cst.Call)
  receiver: cst.BaseExpression | None = _get_receiver(node)
  assert isinstance(receiver, cst.Name)
  assert receiver.value == "self"
  node2 = cst.ensure_type(cst.parse_expression("register_buffer('name', tensor)"), cst.Call)
  receiver2: cst.BaseExpression | None = _get_receiver(node2)
  assert receiver2 is None


def test_convert_register_buffer() -> None:
  """Converts register buffer."""
  node = cst.ensure_type(cst.parse_expression("self.register_buffer('name')"), cst.Call)
  ctx: DummyContext = DummyContext({"BatchStat": "flax.nnx.BatchStat"})
  assert getattr(
    {
      "v": getattr({"v": convert_register_buffer}, "__getitem__")("v")(
        getattr({"v": node}, "__getitem__")("v"), getattr({"v": ctx}, "__getitem__")("v")
      )
    },
    "__getitem__",
  )("v") is getattr({"v": node}, "__getitem__")("v")
  node2 = cst.ensure_type(cst.parse_expression("register_buffer('name', t)"), cst.Call)
  assert getattr(
    {
      "v": getattr({"v": convert_register_buffer}, "__getitem__")("v")(
        getattr({"v": node2}, "__getitem__")("v"), getattr({"v": ctx}, "__getitem__")("v")
      )
    },
    "__getitem__",
  )("v") is getattr({"v": node2}, "__getitem__")("v")
  node3 = cst.ensure_type(cst.parse_expression("self.register_buffer('name', t)"), cst.Call)
  ctx_empty: DummyContext = DummyContext()
  assert getattr(
    {
      "v": getattr({"v": convert_register_buffer}, "__getitem__")("v")(
        getattr({"v": node3}, "__getitem__")("v"), getattr({"v": ctx_empty}, "__getitem__")("v")
      )
    },
    "__getitem__",
  )("v") is getattr({"v": node3}, "__getitem__")("v")
  node4 = cst.ensure_type(cst.parse_expression("self.register_buffer('name', t)"), cst.Call)
  ctx_batchstat: DummyContext = DummyContext({"BatchStat": "flax.nnx.BatchStat"})
  result: cst.CSTNode | cst.Call = getattr(
    {
      "v": getattr({"v": convert_register_buffer}, "__getitem__")("v")(
        getattr({"v": node4}, "__getitem__")("v"), getattr({"v": ctx_batchstat}, "__getitem__")("v")
      )
    },
    "__getitem__",
  )("v")
  assert isinstance(result, cst.Call)
  assert isinstance(result.func, cst.Name)
  assert result.func.value == "setattr"
  assert len(result.args) == 3
  wrapper_call: cst.BaseExpression = result.args[2].value
  assert isinstance(wrapper_call, cst.Call)
  assert isinstance(wrapper_call.func, cst.Attribute)
  assert wrapper_call.func.attr.value == "BatchStat"
  assert isinstance(wrapper_call.func.value, cst.Attribute)
  assert wrapper_call.func.value.attr.value == "nnx"


def test_convert_register_parameter() -> None:
  """Converts register parameter."""
  node = cst.ensure_type(cst.parse_expression("self.register_parameter('name')"), cst.Call)
  ctx: DummyContext = DummyContext({"Param": "flax.nnx.Param"})
  assert getattr(
    {
      "v": getattr({"v": convert_register_parameter}, "__getitem__")("v")(
        getattr({"v": node}, "__getitem__")("v"), getattr({"v": ctx}, "__getitem__")("v")
      )
    },
    "__getitem__",
  )("v") is getattr({"v": node}, "__getitem__")("v")
  node2 = cst.ensure_type(cst.parse_expression("register_parameter('name', p)"), cst.Call)
  assert getattr(
    {
      "v": getattr({"v": convert_register_parameter}, "__getitem__")("v")(
        getattr({"v": node2}, "__getitem__")("v"), getattr({"v": ctx}, "__getitem__")("v")
      )
    },
    "__getitem__",
  )("v") is getattr({"v": node2}, "__getitem__")("v")
  node3 = cst.ensure_type(cst.parse_expression("self.register_parameter('name', p)"), cst.Call)
  ctx_empty: DummyContext = DummyContext()
  assert getattr(
    {
      "v": getattr({"v": convert_register_parameter}, "__getitem__")("v")(
        getattr({"v": node3}, "__getitem__")("v"), getattr({"v": ctx_empty}, "__getitem__")("v")
      )
    },
    "__getitem__",
  )("v") is getattr({"v": node3}, "__getitem__")("v")
  node4 = cst.ensure_type(cst.parse_expression("self.register_parameter('name', p)"), cst.Call)
  ctx_param: DummyContext = DummyContext({"Param": "flax.nnx.Param"})
  result: cst.CSTNode | cst.Call = getattr(
    {
      "v": getattr({"v": convert_register_parameter}, "__getitem__")("v")(
        getattr({"v": node4}, "__getitem__")("v"), getattr({"v": ctx_param}, "__getitem__")("v")
      )
    },
    "__getitem__",
  )("v")
  assert isinstance(result, cst.Call)
  assert isinstance(result.func, cst.Name)
  assert result.func.value == "setattr"
  assert len(result.args) == 3


def test_convert_state_dict() -> None:
  """Converts state dictionary."""
  node = cst.ensure_type(cst.parse_expression("state_dict()"), cst.Call)
  ctx: DummyContext = DummyContext({"ModuleState": "flax.nnx.state"})
  assert getattr(
    {
      "v": getattr({"v": convert_state_dict}, "__getitem__")("v")(
        getattr({"v": node}, "__getitem__")("v"), getattr({"v": ctx}, "__getitem__")("v")
      )
    },
    "__getitem__",
  )("v") is getattr({"v": node}, "__getitem__")("v")
  node2 = cst.ensure_type(cst.parse_expression("self.state_dict()"), cst.Call)
  ctx_empty: DummyContext = DummyContext()
  assert getattr(
    {
      "v": getattr({"v": convert_state_dict}, "__getitem__")("v")(
        getattr({"v": node2}, "__getitem__")("v"), getattr({"v": ctx_empty}, "__getitem__")("v")
      )
    },
    "__getitem__",
  )("v") is getattr({"v": node2}, "__getitem__")("v")
  node3 = cst.ensure_type(cst.parse_expression("model.state_dict()"), cst.Call)
  ctx_state: DummyContext = DummyContext({"ModuleState": "flax.nnx.state"})
  result: cst.CSTNode | cst.Call = getattr(
    {
      "v": getattr({"v": convert_state_dict}, "__getitem__")("v")(
        getattr({"v": node3}, "__getitem__")("v"), getattr({"v": ctx_state}, "__getitem__")("v")
      )
    },
    "__getitem__",
  )("v")
  assert isinstance(result, cst.Call)
  assert isinstance(result.func, cst.Attribute)
  assert result.func.attr.value == "to_pure_dict"
  inner_call: cst.BaseExpression = result.func.value
  assert isinstance(inner_call, cst.Call)
  assert isinstance(inner_call.func, cst.Attribute)
  assert inner_call.func.attr.value == "state"


def test_convert_load_state_dict() -> None:
  """Converts load state dictionary."""
  node = cst.ensure_type(cst.parse_expression("load_state_dict(sd)"), cst.Call)
  ctx: DummyContext = DummyContext({"UpdateState": "flax.nnx.update"})
  assert getattr(
    {
      "v": getattr({"v": convert_load_state_dict}, "__getitem__")("v")(
        getattr({"v": node}, "__getitem__")("v"), getattr({"v": ctx}, "__getitem__")("v")
      )
    },
    "__getitem__",
  )("v") is getattr({"v": node}, "__getitem__")("v")
  node2 = cst.ensure_type(cst.parse_expression("self.load_state_dict()"), cst.Call)
  assert getattr(
    {
      "v": getattr({"v": convert_load_state_dict}, "__getitem__")("v")(
        getattr({"v": node2}, "__getitem__")("v"), getattr({"v": ctx}, "__getitem__")("v")
      )
    },
    "__getitem__",
  )("v") is getattr({"v": node2}, "__getitem__")("v")
  node3 = cst.ensure_type(cst.parse_expression("self.load_state_dict(sd)"), cst.Call)
  ctx_empty: DummyContext = DummyContext()
  assert getattr(
    {
      "v": getattr({"v": convert_load_state_dict}, "__getitem__")("v")(
        getattr({"v": node3}, "__getitem__")("v"), getattr({"v": ctx_empty}, "__getitem__")("v")
      )
    },
    "__getitem__",
  )("v") is getattr({"v": node3}, "__getitem__")("v")
  node4 = cst.ensure_type(cst.parse_expression("model.load_state_dict(sd)"), cst.Call)
  ctx_update: DummyContext = DummyContext({"UpdateState": "flax.nnx.update"})
  result: cst.CSTNode | cst.Call = getattr(
    {
      "v": getattr({"v": convert_load_state_dict}, "__getitem__")("v")(
        getattr({"v": node4}, "__getitem__")("v"), getattr({"v": ctx_update}, "__getitem__")("v")
      )
    },
    "__getitem__",
  )("v")
  assert isinstance(result, cst.Call)
  assert isinstance(result.func, cst.Attribute)
  assert result.func.attr.value == "update"
  assert len(result.args) == 2


def test_convert_parameters() -> None:
  """Converts parameters."""
  node = cst.ensure_type(cst.parse_expression("parameters()"), cst.Call)
  ctx: DummyContext = DummyContext({"ModuleState": "flax.nnx.state", "Param": "flax.nnx.Param"})
  assert getattr(
    {
      "v": getattr({"v": convert_parameters}, "__getitem__")("v")(
        getattr({"v": node}, "__getitem__")("v"), getattr({"v": ctx}, "__getitem__")("v")
      )
    },
    "__getitem__",
  )("v") is getattr({"v": node}, "__getitem__")("v")
  node2 = cst.ensure_type(cst.parse_expression("model.parameters()"), cst.Call)
  ctx_missing_state: DummyContext = DummyContext({"Param": "flax.nnx.Param"})
  assert getattr(
    {
      "v": getattr({"v": convert_parameters}, "__getitem__")("v")(
        getattr({"v": node2}, "__getitem__")("v"), getattr({"v": ctx_missing_state}, "__getitem__")("v")
      )
    },
    "__getitem__",
  )("v") is getattr({"v": node2}, "__getitem__")("v")
  ctx_missing_param: DummyContext = DummyContext({"ModuleState": "flax.nnx.state"})
  assert getattr(
    {
      "v": getattr({"v": convert_parameters}, "__getitem__")("v")(
        getattr({"v": node2}, "__getitem__")("v"), getattr({"v": ctx_missing_param}, "__getitem__")("v")
      )
    },
    "__getitem__",
  )("v") is getattr({"v": node2}, "__getitem__")("v")
  node3 = cst.ensure_type(cst.parse_expression("model.parameters()"), cst.Call)
  ctx_both: DummyContext = DummyContext({"ModuleState": "flax.nnx.state", "Param": "flax.nnx.Param"})
  result: cst.CSTNode | cst.Call = getattr(
    {
      "v": getattr({"v": convert_parameters}, "__getitem__")("v")(
        getattr({"v": node3}, "__getitem__")("v"), getattr({"v": ctx_both}, "__getitem__")("v")
      )
    },
    "__getitem__",
  )("v")
  assert isinstance(result, cst.Call)
  assert isinstance(result.func, cst.Attribute)
  assert result.func.attr.value == "values"
  state_call: cst.BaseExpression = result.func.value
  assert isinstance(state_call, cst.Call)
  assert isinstance(state_call.func, cst.Attribute)
  assert state_call.func.attr.value == "state"
  assert len(state_call.args) == 2
