"""
Plugin for handling Stateful Container Logic.

Handles mapping of container management methods between frameworks, particularly
impedance mismatches between PyTorch's imperative `register_buffer`/`parameters` system
and Flax NNX's explicit state management.

Mappings (Torch -> JAX/NNX):

1. `self.register_buffer("name", t)` -> `setattr(self, "name", flax.nnx.BatchStat(t))`
2. `self.register_parameter("name", p)` -> `setattr(self, "name", flax.nnx.Param(p))`
3. `model.state_dict()` -> `flax.nnx.state(model).to_pure_dict()`
4. `model.load_state_dict(sd)` -> `flax.nnx.update(model, sd)`
5. `model.parameters()` -> `flax.nnx.state(model, flax.nnx.Param).values()`
"""

import libcst as cst
from ml_switcheroo.core.hooks import register_hook, HookContext


def _create_node(code: str) -> cst.BaseExpression:
  """Helper to parse a simple expression string into a CST node."""
  return cst.parse_expression(code)


@register_hook("torch_register_buffer_to_nnx")
def convert_register_buffer(node: cst.Call, ctx: HookContext) -> cst.Call:
  """
  Transforms `self.register_buffer('name', tensor)` -> `setattr(self, 'name', nnx.BatchStat(tensor))`.
  """
  # Check args: expected (name, tensor)
  if len(node.args) < 2:
    return node

  name_arg = node.args[0].value
  tensor_arg = node.args[1].value

  # Check receiver (usually 'self')
  receiver = None
  if isinstance(node.func, cst.Attribute):
    receiver = node.func.value

  if not receiver:
    return node

  # Construct setattr(receiver, name, nnx.BatchStat(tensor))

  # 1. Wrap tensor in BatchStat
  # nnx.BatchStat(tensor)
  batch_stat_call = cst.Call(func=_create_node("flax.nnx.BatchStat"), args=[cst.Arg(value=tensor_arg)])

  # 2. Construct setattr
  new_call = cst.Call(
    func=cst.Name("setattr"),
    args=[
      cst.Arg(value=receiver, comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" "))),
      cst.Arg(value=name_arg, comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" "))),
      cst.Arg(value=batch_stat_call),
    ],
  )

  return new_call


@register_hook("torch_register_parameter_to_nnx")
def convert_register_parameter(node: cst.Call, ctx: HookContext) -> cst.Call:
  """
  Transforms `self.register_parameter('name', param)` -> `setattr(self, 'name', nnx.Param(param))`.
  """
  if len(node.args) < 2:
    return node

  name_arg = node.args[0].value
  tensor_arg = node.args[1].value

  receiver = node.func.value if isinstance(node.func, cst.Attribute) else None
  if not receiver:
    return node

  # nnx.Param(tensor)
  param_call = cst.Call(func=_create_node("flax.nnx.Param"), args=[cst.Arg(value=tensor_arg)])

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
  Transforms `model.state_dict()` -> `flax.nnx.state(model).to_pure_dict()`.
  """
  receiver = node.func.value if isinstance(node.func, cst.Attribute) else None
  if not receiver:
    return node

  # Pattern: flax.nnx.state(model).to_pure_dict()
  # 1. flax.nnx.state(model)
  inner_call = cst.Call(func=_create_node("flax.nnx.state"), args=[cst.Arg(value=receiver)])

  # 2. .to_pure_dict()
  outer_call = cst.Call(func=cst.Attribute(value=inner_call, attr=cst.Name("to_pure_dict")), args=[])
  return outer_call


@register_hook("torch_load_state_dict_to_nnx")
def convert_load_state_dict(node: cst.Call, ctx: HookContext) -> cst.Call:
  """
  Transforms `model.load_state_dict(state)` -> `flax.nnx.update(model, state)`.
  """
  receiver = node.func.value if isinstance(node.func, cst.Attribute) else None
  if not receiver:
    return node

  if not node.args:
    return node
  state_arg = node.args[0].value

  # flax.nnx.update(model, state)
  new_call = cst.Call(
    func=_create_node("flax.nnx.update"),
    args=[
      cst.Arg(value=receiver, comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" "))),
      cst.Arg(value=state_arg),
    ],
  )
  return new_call


@register_hook("torch_parameters_to_nnx")
def convert_parameters(node: cst.Call, ctx: HookContext) -> cst.Call:
  """
  Transforms `model.parameters()` -> `flax.nnx.state(model, flax.nnx.Param).values()`.
  """
  receiver = node.func.value if isinstance(node.func, cst.Attribute) else None
  if not receiver:
    return node

  # flax.nnx.state(model, flax.nnx.Param)
  state_call = cst.Call(
    func=_create_node("flax.nnx.state"),
    args=[
      cst.Arg(value=receiver, comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" "))),
      cst.Arg(value=_create_node("flax.nnx.Param")),
    ],
  )

  # .values()
  final_call = cst.Call(func=cst.Attribute(value=state_call, attr=cst.Name("values")), args=[])
  return final_call
