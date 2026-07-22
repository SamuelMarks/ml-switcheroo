"""Test suite for the State Container module."""

import libcst as cst
from unittest.mock import patch
from ml_switcheroo.plugins.state_container import (
  _create_node,
  _get_receiver,
  convert_register_buffer,
  convert_register_parameter,
  convert_state_dict,
  convert_load_state_dict,
  convert_parameters,
)


class DummyContext:
  """Dummy Context class for testing purposes."""

  def __init__(self, api_map=None):
    """Initializes the DummyContext instance."""
    self.api_map = api_map or {}

  def lookup_api(self, op_id):
    """Mock implementation of lookup API."""
    return self.api_map.get(op_id)


def test_create_node():
  """Creates node."""
  node = _create_node("a.b.c")
  assert isinstance(node, cst.Attribute)
  with patch("libcst.parse_expression", side_effect=Exception("Failed")):
    node = _create_node("fallback_name")
  assert isinstance(node, cst.Name)
  assert node.value == "fallback_name"


def test_get_receiver():
  """Gets receiver."""
  node = cst.parse_expression("self.register_buffer('name', tensor)")
  receiver = _get_receiver(node)
  assert isinstance(receiver, cst.Name)
  assert receiver.value == "self"
  node = cst.parse_expression("register_buffer('name', tensor)")
  receiver = _get_receiver(node)
  assert receiver is None


def test_convert_register_buffer():
  """Converts register buffer."""
  node = cst.parse_expression("self.register_buffer('name')")
  ctx = DummyContext({"BatchStat": "flax.nnx.BatchStat"})
  assert convert_register_buffer(node, ctx) is node
  node = cst.parse_expression("register_buffer('name', t)")
  assert convert_register_buffer(node, ctx) is node
  node = cst.parse_expression("self.register_buffer('name', t)")
  ctx_empty = DummyContext()
  assert convert_register_buffer(node, ctx_empty) is node
  node = cst.parse_expression("self.register_buffer('name', t)")
  ctx = DummyContext({"BatchStat": "flax.nnx.BatchStat"})
  result = convert_register_buffer(node, ctx)
  assert isinstance(result, cst.Call)
  assert result.func.value == "setattr"
  assert len(result.args) == 3
  wrapper_call = result.args[2].value
  assert isinstance(wrapper_call, cst.Call)
  assert wrapper_call.func.attr.value == "BatchStat"
  assert wrapper_call.func.value.attr.value == "nnx"


def test_convert_register_parameter():
  """Converts register parameter."""
  node = cst.parse_expression("self.register_parameter('name')")
  ctx = DummyContext({"Param": "flax.nnx.Param"})
  assert convert_register_parameter(node, ctx) is node
  node = cst.parse_expression("register_parameter('name', p)")
  assert convert_register_parameter(node, ctx) is node
  node = cst.parse_expression("self.register_parameter('name', p)")
  ctx_empty = DummyContext()
  assert convert_register_parameter(node, ctx_empty) is node
  node = cst.parse_expression("self.register_parameter('name', p)")
  ctx = DummyContext({"Param": "flax.nnx.Param"})
  result = convert_register_parameter(node, ctx)
  assert isinstance(result, cst.Call)
  assert result.func.value == "setattr"
  assert len(result.args) == 3


def test_convert_state_dict():
  """Converts state dictionary."""
  node = cst.parse_expression("state_dict()")
  ctx = DummyContext({"ModuleState": "flax.nnx.state"})
  assert convert_state_dict(node, ctx) is node
  node = cst.parse_expression("self.state_dict()")
  ctx_empty = DummyContext()
  assert convert_state_dict(node, ctx_empty) is node
  node = cst.parse_expression("model.state_dict()")
  ctx = DummyContext({"ModuleState": "flax.nnx.state"})
  result = convert_state_dict(node, ctx)
  assert isinstance(result, cst.Call)
  assert result.func.attr.value == "to_pure_dict"
  inner_call = result.func.value
  assert isinstance(inner_call, cst.Call)
  assert inner_call.func.attr.value == "state"


def test_convert_load_state_dict():
  """Converts load state dictionary."""
  node = cst.parse_expression("load_state_dict(sd)")
  ctx = DummyContext({"UpdateState": "flax.nnx.update"})
  assert convert_load_state_dict(node, ctx) is node
  node = cst.parse_expression("self.load_state_dict()")
  assert convert_load_state_dict(node, ctx) is node
  node = cst.parse_expression("self.load_state_dict(sd)")
  ctx_empty = DummyContext()
  assert convert_load_state_dict(node, ctx_empty) is node
  node = cst.parse_expression("model.load_state_dict(sd)")
  ctx = DummyContext({"UpdateState": "flax.nnx.update"})
  result = convert_load_state_dict(node, ctx)
  assert isinstance(result, cst.Call)
  assert result.func.attr.value == "update"
  assert len(result.args) == 2


def test_convert_parameters():
  """Converts parameters."""
  node = cst.parse_expression("parameters()")
  ctx = DummyContext({"ModuleState": "flax.nnx.state", "Param": "flax.nnx.Param"})
  assert convert_parameters(node, ctx) is node
  node = cst.parse_expression("model.parameters()")
  ctx_missing_state = DummyContext({"Param": "flax.nnx.Param"})
  assert convert_parameters(node, ctx_missing_state) is node
  ctx_missing_param = DummyContext({"ModuleState": "flax.nnx.state"})
  assert convert_parameters(node, ctx_missing_param) is node
  node = cst.parse_expression("model.parameters()")
  ctx = DummyContext({"ModuleState": "flax.nnx.state", "Param": "flax.nnx.Param"})
  result = convert_parameters(node, ctx)
  assert isinstance(result, cst.Call)
  assert result.func.attr.value == "values"
  state_call = result.func.value
  assert isinstance(state_call, cst.Call)
  assert state_call.func.attr.value == "state"
  assert len(state_call.args) == 2
