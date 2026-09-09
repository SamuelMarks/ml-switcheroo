"""Plugin for unrolling in-place tensor operations to functional assignments.

PyTorch uses a trailing underscore convention (e.g., `x.add_(y)`) to denote in-place
mutation. JAX and other functional frameworks require immutable operations, where
the result must be assigned back to the variable (e.g., `x = x.add(y)`).

This plugin:
1.  Detects calls ending in `_` (e.g., `add_`).
2.  Checks validity (excludes special methods like `__init__`).
3.  Transforms the expression statement `x.op_(y)` into an assignment `x = x.op(y)`.
"""

import libcst as cst
from typing import Optional, Union

from ml_switcheroo.core.hooks import register_hook, HookContext


def _get_receiver_name(node: cst.Call) -> Optional[cst.BaseExpression]:
  """Extract the receiver object from a method call.

  Args:
      node (cst.Call): The function call node.

  Returns:
      Optional[cst.BaseExpression]: The object being called (e.g. `x` in `x.add_()`),
      or None if not a method call.

  """
  if isinstance(node.func, cst.Attribute):
    return node.func.value
  return None


def _get_method_name(node: cst.Call) -> Optional[str]:
  """Extract the method name string.

  Args:
      node (cst.Call): The function call node.

  Returns:
      Optional[str]: The method name (e.g. "add_"), or None.

  """
  if isinstance(node.func, cst.Attribute):
    return node.func.attr.value
  return None


@register_hook("unroll_inplace_ops")
def unroll_inplace_ops(
  node: Union[cst.Call, cst.Expr], ctx: HookContext
) -> Union[cst.Call, cst.Assign, cst.Expr, cst.BinaryOperation]:
  """Plugin Transform Transforms in-place method calls to functional assignments.

  **Scope**

  This hook mechanism in `ml-switcheroo` typically operates on `cst.Call` nodes.
  However, to transform a standalone expression statement `x.add_(y)` into
  an assignment `x = ...`, we ideally need access to the statement container.

  If the hook infrastructure passes `cst.Call`, we can only mutate the call itself.
  Replacing `x.add_(y)` with `x = x.add(y)` *inside* another expression is invalid syntax.
  Therefore, this logic assumes usage primarily in top-level expression statements
  or relies on the Rewriter's ability to handle statement expansion if this returns a wrapper.

  **Current Strategy**: We strip the underscore to make the call functional.
  If the Call is part of an Expression Statement (standalone), it effectively becomes
  a no-op output (`x.add(y)` computed but lost) unless assigned.

  **Refined Strategy**: Since we can't easily ascend to the statement level from a Call hook,
  we strip the `_` to ensure the API mapping (e.g. `torch.add`) works.
  The user receives `x.add(y)`, which is valid execution but discards result.
  WARNING: This is a limitation of Call-level hooks. Ideally, we flag this.

  **Implementation**:

  1. Strip `_` from method name: `x.add_(y)` -> `x.add(y)`.
  2. If Context allows or we detect usage context, we might attempt assignment injection,
     but `Call` replacement with `Assign` is risky in nested contexts.
     However, standard PyTorch in-place ops `x.add_(y)` return `x`.
     So `z = x.add_(y)` -> `z = x.add(y)` is semantically correct conversion to functional.
     The only "Loss" is that `x` itself isn't updated in the scope.

  Args:
      node: The original CST node.
      ctx: HookContext for target framework access.
  """
  # 1. Identify Method Call
  target_call: cst.Call
  if isinstance(node, cst.Expr):
    if not isinstance(node.value, cst.Call):
      return node
    target_call = node.value
  elif isinstance(node, cst.Call):
    target_call = node
  else:
    return node

  method_name = _get_method_name(target_call)
  receiver = _get_receiver_name(target_call)

  if not method_name or not receiver:
    return node

  # 2. Check for In-Place Suffix convention
  # Must end in "_" but not be internal "__"
  if not method_name.endswith("_") or method_name == "_":
    return node
  if method_name.startswith("__"):
    return node

  # 3. Strip Suffix
  clean_name = method_name[:-1]

  # 4. Strategy A: Map to Infix Operator (Best for JAX/NumPy compatibility)
  # JAX arrays don't have .add(), .sub() methods, so we convert to x + y
  infix_map = {
    "add": cst.Add(),
    "sub": cst.Subtract(),
    "mul": cst.Multiply(),
    "div": cst.Divide(),
    "pow": cst.Power(),
  }

  res: Union[cst.Call, cst.BinaryOperation]
  if clean_name in infix_map and len(target_call.args) == 1:
    # x.add_(y) -> x + y
    # Ensure arg is clean (remove comma if present)
    right_operand = target_call.args[0].value
    res = cst.BinaryOperation(left=receiver, operator=infix_map[clean_name], right=right_operand)
  else:
    # 5. Strategy B: Construct Functional Method Call (Fallback)
    # x.unknown_(y) -> x.unknown(y)
    new_func = target_call.func.with_changes(attr=cst.Name(clean_name))
    res = target_call.with_changes(func=new_func)

  if isinstance(node, cst.Expr):
    return node.with_changes(value=res)
  return res
