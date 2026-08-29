"""Test suite for the State Container module."""

from typing import Dict, Optional, Union
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
  """Docstring."""

  def __init__(self, api_map: Optional[Dict[str, str]] = None) -> None:
    """Initializes the DummyContext instance."""
    self.api_map: Dict[str, str] = api_map or {}

  def lookup_api(self, op_id: str) -> Optional[str]:
    """Mock implementation of lookup API."""
    return self.api_map.get(op_id)


def test_create_node() -> None:
  """Creates node."""
  node: Union[cst.Name, cst.Attribute] = _create_node("a.b.c")
  assert isinstance(node, cst.Attribute)
  with patch("libcst.parse_expression", side_effect=Exception("Failed")):
    node = _create_node("fallback_name")
  assert isinstance(node, cst.Name)
  assert node.value == "fallback_name"


def test_get_receiver() -> None:
  """Gets receiver."""
  node: cst.Call = cst.parse_expression("self.register_buffer('name', tensor)")
  receiver: Optional[cst.BaseExpression] = _get_receiver(node)
  assert isinstance(receiver, cst.Name)
  assert receiver.value == "self"
  node2: cst.Call = cst.parse_expression("register_buffer('name', tensor)")
  receiver2: Optional[cst.BaseExpression] = _get_receiver(node2)
  assert receiver2 is None


def test_convert_register_buffer() -> None:
  """Converts register buffer."""
  node: cst.Call = cst.parse_expression("self.register_buffer('name')")
  ctx: DummyContext = DummyContext({"BatchStat": "flax.nnx.BatchStat"})
  assert convert_register_buffer(node, ctx) is node
  node2: cst.Call = cst.parse_expression("register_buffer('name', t)")
  assert convert_register_buffer(node2, ctx) is node2
  node3: cst.Call = cst.parse_expression("self.register_buffer('name', t)")
  ctx_empty: DummyContext = DummyContext()
  assert convert_register_buffer(node3, ctx_empty) is node3
  node4: cst.Call = cst.parse_expression("self.register_buffer('name', t)")
  ctx_batchstat: DummyContext = DummyContext({"BatchStat": "flax.nnx.BatchStat"})
  result: Union[cst.CSTNode, cst.Call] = convert_register_buffer(node4, ctx_batchstat)
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
  node: cst.Call = cst.parse_expression("self.register_parameter('name')")
  ctx: DummyContext = DummyContext({"Param": "flax.nnx.Param"})
  assert convert_register_parameter(node, ctx) is node
  node2: cst.Call = cst.parse_expression("register_parameter('name', p)")
  assert convert_register_parameter(node2, ctx) is node2
  node3: cst.Call = cst.parse_expression("self.register_parameter('name', p)")
  ctx_empty: DummyContext = DummyContext()
  assert convert_register_parameter(node3, ctx_empty) is node3
  node4: cst.Call = cst.parse_expression("self.register_parameter('name', p)")
  ctx_param: DummyContext = DummyContext({"Param": "flax.nnx.Param"})
  result: Union[cst.CSTNode, cst.Call] = convert_register_parameter(node4, ctx_param)
  assert isinstance(result, cst.Call)
  assert isinstance(result.func, cst.Name)
  assert result.func.value == "setattr"
  assert len(result.args) == 3


def test_convert_state_dict() -> None:
  """Converts state dictionary."""
  node: cst.Call = cst.parse_expression("state_dict()")
  ctx: DummyContext = DummyContext({"ModuleState": "flax.nnx.state"})
  assert convert_state_dict(node, ctx) is node
  node2: cst.Call = cst.parse_expression("self.state_dict()")
  ctx_empty: DummyContext = DummyContext()
  assert convert_state_dict(node2, ctx_empty) is node2
  node3: cst.Call = cst.parse_expression("model.state_dict()")
  ctx_state: DummyContext = DummyContext({"ModuleState": "flax.nnx.state"})
  result: Union[cst.CSTNode, cst.Call] = convert_state_dict(node3, ctx_state)
  assert isinstance(result, cst.Call)
  assert isinstance(result.func, cst.Attribute)
  assert result.func.attr.value == "to_pure_dict"
  inner_call: cst.BaseExpression = result.func.value
  assert isinstance(inner_call, cst.Call)
  assert isinstance(inner_call.func, cst.Attribute)
  assert inner_call.func.attr.value == "state"


def test_convert_load_state_dict() -> None:
  """Converts load state dictionary."""
  node: cst.Call = cst.parse_expression("load_state_dict(sd)")
  ctx: DummyContext = DummyContext({"UpdateState": "flax.nnx.update"})
  assert convert_load_state_dict(node, ctx) is node
  node2: cst.Call = cst.parse_expression("self.load_state_dict()")
  assert convert_load_state_dict(node2, ctx) is node2
  node3: cst.Call = cst.parse_expression("self.load_state_dict(sd)")
  ctx_empty: DummyContext = DummyContext()
  assert convert_load_state_dict(node3, ctx_empty) is node3
  node4: cst.Call = cst.parse_expression("model.load_state_dict(sd)")
  ctx_update: DummyContext = DummyContext({"UpdateState": "flax.nnx.update"})
  result: Union[cst.CSTNode, cst.Call] = convert_load_state_dict(node4, ctx_update)
  assert isinstance(result, cst.Call)
  assert isinstance(result.func, cst.Attribute)
  assert result.func.attr.value == "update"
  assert len(result.args) == 2


def test_convert_parameters() -> None:
  """Converts parameters."""
  node: cst.Call = cst.parse_expression("parameters()")
  ctx: DummyContext = DummyContext({"ModuleState": "flax.nnx.state", "Param": "flax.nnx.Param"})
  assert convert_parameters(node, ctx) is node
  node2: cst.Call = cst.parse_expression("model.parameters()")
  ctx_missing_state: DummyContext = DummyContext({"Param": "flax.nnx.Param"})
  assert convert_parameters(node2, ctx_missing_state) is node2
  ctx_missing_param: DummyContext = DummyContext({"ModuleState": "flax.nnx.state"})
  assert convert_parameters(node2, ctx_missing_param) is node2
  node3: cst.Call = cst.parse_expression("model.parameters()")
  ctx_both: DummyContext = DummyContext({"ModuleState": "flax.nnx.state", "Param": "flax.nnx.Param"})
  result: Union[cst.CSTNode, cst.Call] = convert_parameters(node3, ctx_both)
  assert isinstance(result, cst.Call)
  assert isinstance(result.func, cst.Attribute)
  assert result.func.attr.value == "values"
  state_call: cst.BaseExpression = result.func.value
  assert isinstance(state_call, cst.Call)
  assert isinstance(state_call.func, cst.Attribute)
  assert state_call.func.attr.value == "state"
  assert len(state_call.args) == 2
