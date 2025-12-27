"""
Plugin for MLX Ecosystem Mapping.

Handles:
1. Compilation: `@torch.compile` -> `@mx.compile`.
2. Eager Evaluation: `torch.cuda.synchronize()` -> Warning/No-op.
"""

import libcst as cst
from typing import Union

from ml_switcheroo.core.hooks import register_hook, HookContext


def _create_dotted_name(name_str: str) -> cst.BaseExpression:
  """Helper: Creates a CST Attribute chain from string."""
  parts = name_str.split(".")
  node = cst.Name(parts[0])
  for part in parts[1:]:
    node = cst.Attribute(value=node, attr=cst.Name(part))
  return node


@register_hook("mlx_compiler")
def transform_compiler(node: Union[cst.Decorator, cst.Call], ctx: HookContext) -> cst.CSTNode:
  """
  Hook: Maps JIT compilation decorators.

  Triggers: Operations mapped with `requires_plugin: "mlx_compiler"`.

  Transformation:
      Input:  `@torch.compile(fullgraph=True, dynamic=True)`
      Output: `@mx.compile` (stripping incompatible kwargs).

  Decoupling:
      Looks up the `Compile` operation API in semantics (e.g. `mlx.core.compile`).
  """
  # Resolve Target API dynamically
  target_api = ctx.lookup_api("Compile") or "mlx.core.compile"
  new_func = _create_dotted_name(target_api)

  # Validating Input Type
  # Is it a Decorator node?
  if hasattr(node, "decorator"):
    # Strip arguments completely because Torch kwargs (backend="inductor")
    # are invalid in MLX.
    # Return decorator with just the function name (no Call node)
    return node.with_changes(decorator=new_func)

  # If it's a Call node (e.g. used as functional wrapper `c_fn = torch.compile(fn)`),
  # strip kwargs and just return `mx.compile(fn)`.
  if isinstance(node, cst.Call):
    # Keep positional arg 0 (the function)
    new_args = []
    if node.args:
      new_args.append(node.args[0])
    return node.with_changes(func=new_func, args=new_args)

  return node


@register_hook("mlx_synchronize")
def transform_synchronize(node: cst.Call, ctx: HookContext) -> cst.CSTNode:
  """
  Hook: Maps barrier synchronization to a warning.

  MLX is lazy, but `torch.cuda.synchronize()` implies a global device barrier.
  Equivalent `mx.eval()` requires arguments. Since we cannot infer state variables here,
  we replace the call with a print statement to alert the user.
  """
  message_str = "# [Switcheroo] Global sync requires explicit tensor args in target framework."

  # Return a print call: print("...")
  return cst.Call(func=cst.Name("print"), args=[cst.Arg(value=cst.SimpleString(f"'{message_str}'"))])
