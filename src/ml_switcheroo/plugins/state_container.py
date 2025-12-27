"""
Plugin for handling Stateful Container Logic.

Handles mapping of container management methods between frameworks, particularly
impedance mismatches between imperative usage (PyTorch's `register_buffer`) and
functional explicit state management (JAX/Flax/MLX).

This module converts:
1. `self.register_buffer("name", t)` -> `setattr(self, "name", Wrapper(t))`
2. `self.register_parameter("name", p)` -> `setattr(self, "name", ParamWrapper(p))`
3. `model.state_dict()` -> `StateFunc(model).to_pure_dict()`
4. `model.load_state_dict(sd)` -> `UpdateFunc(model, sd)`
5. `model.parameters()` -> `StateFunc(model, ParamWrapper).values()`

Decoupling:
    The specific wrapper definitions (e.g. `flax.nnx.BatchStat` or `custom.State`)
    must be defined in the Semantic Knowledge Base. Lookups are strict; if no
    mapping exists for the Abstract Operation (e.g. 'BatchStat'), the hook
    aborts and preserves the original code.
"""

import libcst as cst
from typing import Optional, List
from ml_switcheroo.core.hooks import register_hook, HookContext


def _create_node(code: str) -> cst.BaseExpression:
  """Helper to parse a simple expression string into a CST node."""
  try:
    return cst.parse_expression(code)
  except Exception:
    # Fallback for simple identifiers if expression parsing fails
    return cst.Name(code)


def _get_receiver(node: cst.Call) -> Optional[cst.BaseExpression]:
  """Helper to extract the object instance being called (e.g. 'self' or 'model')."""
  if isinstance(node.func, cst.Attribute):
    return node.func.value
  return None


@register_hook("torch_register_buffer_to_nnx")
def convert_register_buffer(node: cst.Call, ctx: HookContext) -> cst.Call:
  """
  Transforms `register_buffer`.

  Target: `setattr(self, 'name', Wrapper(tensor))`

  Abstract Op Lookup: "BatchStat"
  """
  # Check args: expected (name, tensor)
  if len(node.args) < 2:
    return node

  name_arg = node.args[0].value
  tensor_arg = node.args[1].value

  receiver = _get_receiver(node)
  if not receiver:
    return node

  # Strict Lookup: Abort if knowledge base doesn't define 'BatchStat' for target
  wrapper_api = ctx.lookup_api("BatchStat")
  if not wrapper_api:
    return node

  # 1. Wrap tensor: Wrapper(tensor)
  wrapper_call = cst.Call(func=_create_node(wrapper_api), args=[cst.Arg(value=tensor_arg)])

  # 2. Construct setattr(receiver, name, wrapper_call)
  new_call = cst.Call(
    func=cst.Name("setattr"),
    args=[
      cst.Arg(value=receiver, comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" "))),
      cst.Arg(value=name_arg, comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" "))),
      cst.Arg(value=wrapper_call),
    ],
  )

  return new_call


@register_hook("torch_register_parameter_to_nnx")
def convert_register_parameter(node: cst.Call, ctx: HookContext) -> cst.Call:
  """
  Transforms `register_parameter`.

  Target: `setattr(self, 'name', ParamWrapper(param))`

  Abstract Op Lookup: "Param"
  """
  if len(node.args) < 2:
    return node

  name_arg = node.args[0].value
  tensor_arg = node.args[1].value

  receiver = _get_receiver(node)
  if not receiver:
    return node

  # Strict Lookup
  wrapper_api = ctx.lookup_api("Param")
  if not wrapper_api:
    return node

  # Wrapper(tensor)
  param_call = cst.Call(func=_create_node(wrapper_api), args=[cst.Arg(value=tensor_arg)])

  new_call = cst.Call(
    func=cst.Name("setattr"),
    args=[
      cst.Arg(value=receiver, comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" "))),
      cst.Arg(value=name_arg, comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" "))),
      cst.Arg(value=param_call),
    ],
  )
  return new_call


@register_hook("torch_state_dict_to_nnx")
def convert_state_dict(node: cst.Call, ctx: HookContext) -> cst.Call:
  """
  Transforms `state_dict`.

  Target: `StateFunc(model).to_pure_dict()`

  Abstract Op Lookup: "ModuleState"
  """
  receiver = _get_receiver(node)
  if not receiver:
    return node

  # Strict Lookup
  state_api = ctx.lookup_api("ModuleState")
  if not state_api:
    return node

  # 1. StateFunc(model)
  inner_call = cst.Call(func=_create_node(state_api), args=[cst.Arg(value=receiver)])

  # 2. .to_pure_dict()
  outer_call = cst.Call(func=cst.Attribute(value=inner_call, attr=cst.Name("to_pure_dict")), args=[])
  return outer_call


@register_hook("torch_load_state_dict_to_nnx")
def convert_load_state_dict(node: cst.Call, ctx: HookContext) -> cst.Call:
  """
  Transforms `load_state_dict`.

  Target: `UpdateFunc(model, state)`

  Abstract Op Lookup: "UpdateState"
  """
  receiver = _get_receiver(node)
  if not receiver:
    return node

  if not node.args:
    return node
  state_arg = node.args[0].value

  # Strict Lookup
  update_api = ctx.lookup_api("UpdateState")
  if not update_api:
    return node

  # UpdateFunc(model, state)
  new_call = cst.Call(
    func=_create_node(update_api),
    args=[
      cst.Arg(value=receiver, comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" "))),
      cst.Arg(value=state_arg),
    ],
  )
  return new_call


@register_hook("torch_parameters_to_nnx")
def convert_parameters(node: cst.Call, ctx: HookContext) -> cst.Call:
  """
  Transforms `parameters`.

  Target: `StateFunc(model, ParamType).values()`

  Abstract Op Lookup: "ModuleState", "Param"
  """
  receiver = _get_receiver(node)
  if not receiver:
    return node

  # Strict Lookup for both required components
  state_api = ctx.lookup_api("ModuleState")
  param_api = ctx.lookup_api("Param")

  if not state_api or not param_api:
    return node

  # StateFunc(model, ParamType)
  state_call = cst.Call(
    func=_create_node(state_api),
    args=[
      cst.Arg(value=receiver, comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" "))),
      cst.Arg(value=_create_node(param_api)),
    ],
  )

  # .values()
  final_call = cst.Call(func=cst.Attribute(value=state_call, attr=cst.Name("values")), args=[])
  return final_call
