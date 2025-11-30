"""
Plugin module defining AST decomposition and recomposition rules.

This module provides hooks to transform complex function calls into simpler
primitives (Decomposition) or reconstruct complex calls from primitives
(Recomposition/Composition) to support bidirectional transpilation.
"""

import libcst as cst

from ml_switcheroo.core.hooks import register_hook, HookContext


def _strip_trailing_comma(args_list: list[cst.Arg]) -> list[cst.Arg]:
  """
  Ensures the last argument in the list has no comma.

  Args:
      args_list: A list of LibCST Argument nodes.

  Returns:
      The modified list with the trailing comma removed from the last item.
  """
  if not args_list:
    return args_list

  last_idx = len(args_list) - 1
  last_arg = args_list[last_idx]

  # Remove the comma from the last argument
  args_list[last_idx] = last_arg.with_changes(comma=cst.MaybeSentinel.DEFAULT)
  return args_list


def _create_dotted_name(api_string: str) -> cst.BaseExpression:
  """
  Creates a CST Attribute chain from a dotted string.

  Args:
      api_string: A dot-separated string (e.g., 'jax.numpy.add').

  Returns:
      A CST node representing the attribute chain (e.g., Attribute(...) or Name(...)).
  """
  parts = api_string.split(".")
  node = cst.Name(parts[0])
  for part in parts[1:]:
    node = cst.Attribute(value=node, attr=cst.Name(part))
  return node


def _resolve_target_name(node: cst.Call, ctx: HookContext, op_name: str) -> cst.BaseExpression:
  """
  Helper to resolve the function name for the target framework.

  Args:
      node: The original Calling node (used for fallback).
      ctx: The hook context containing the semantics manager.
      op_name: The abstract operation name (e.g., 'add').

  Returns:
      A CST BaseExpression for the new function name.
  """
  target_api = ctx.lookup_api(op_name)

  # Fallback to defaults to prevent breakage if user JSONs are incomplete
  if not target_api:
    if ctx.target_fw == "jax":
      target_api = "jax.numpy.add"
    elif ctx.target_fw == "numpy":
      target_api = "numpy.add"
    elif ctx.target_fw == "torch":
      target_api = "torch.add"

  if target_api:
    return _create_dotted_name(target_api)

  # Final Fallback: keep original name if resolution fails
  return node.func


@register_hook("decompose_alpha")
def transform_alpha_add(node: cst.Call, ctx: HookContext) -> cst.Call:
  """
  Transforms an `add` call with an `alpha` parameter into a multiplication.

  Used for converting Torch `add` to JAX/Numpy `add`.

  Transformation:
      Input:  add(x, y, alpha=a)
      Output: jax.numpy.add(x, y * a)

  Args:
      node: The CST Call node to transform.
      ctx: The plugin execution context.

  Returns:
      The transformed CST Call node.
  """
  # 0. Validate Target Capability
  # We are decomposing into a binary add(x, scaled_y).
  # We must ensure the standard definition of 'add' has at least 2 args.
  std_sig = ctx.lookup_signature("add")

  # If standard signature is known but has < 2 args, this decomposition is invalid.
  # If list is empty (unknown op), we proceed boldly (fallback logic).
  if std_sig and len(std_sig) < 2:
    # Cannot decompose to binary add if target only supports unary
    return node

  # 1. Filter and Identify Args
  cleaned_args = []
  alpha_val = None

  for arg in node.args:
    if arg.keyword and arg.keyword.value == "alpha":
      alpha_val = arg.value
    else:
      cleaned_args.append(arg)

  # If no alpha found, we still perform the name swap (identity transform + rename)
  if not alpha_val or len(cleaned_args) < 2:
    new_func = _resolve_target_name(node, ctx, "add")
    return node.with_changes(func=new_func, args=cleaned_args)

  # 2. Modify the target argument (the second one)
  # We assume binary add(x, y), so second arg is being scaled.
  target_arg = cleaned_args[-1]

  # Check parenthesis priority: if alpha is complex, might need parens,
  # but BinaryOperation usually handles precedence safely in LibCST.
  scaled_expr = cst.BinaryOperation(left=target_arg.value, operator=cst.Multiply(), right=alpha_val)

  # Update the arg with the new expression
  cleaned_args[-1] = target_arg.with_changes(value=scaled_expr)

  # 3. Clean Syntax (Remove trailing commas)
  final_args = _strip_trailing_comma(cleaned_args)

  # 4. SWAP FUNCTION NAME
  new_func = _resolve_target_name(node, ctx, "add")

  return node.with_changes(func=new_func, args=final_args)


@register_hook("recompose_alpha")
def transform_alpha_add_reverse(node: cst.Call, ctx: HookContext) -> cst.Call:
  """
  Transforms a multiplication-nested `add` call into an `add` with `alpha`.

  Used for converting JAX/Numpy `add` to Torch `add`.

  Transformation:
      Input:  add(x, y * a)
      Output: torch.add(x, y, alpha=a)

  Args:
      node: The CST Call node to transform.
      ctx: The plugin execution context.

  Returns:
      The transformed CST Call node.
  """
  # 1. Resolve basic name swap first (in case we don't find the pattern)
  new_func = _resolve_target_name(node, ctx, "add")

  # 2. Check strict argument count (must have exactly 2 args to be a candidate)
  if len(node.args) != 2:
    return node.with_changes(func=new_func)

  left_arg = node.args[0]
  right_arg = node.args[1]

  # 3. Check if right_arg is a BinOp (Multiplication)
  # The structure of right_arg is cst.Arg(value=cst.BinaryOperation(...))
  if not isinstance(right_arg.value, cst.BinaryOperation):
    # Pattern mismatch: just return rename
    return node.with_changes(func=new_func)

  bin_op = right_arg.value
  if not isinstance(bin_op.operator, cst.Multiply):
    # Pattern mismatch: just return rename
    return node.with_changes(func=new_func)

  # 4. Extract term and scalar
  # Structure: term * scalar.
  # We assume the right operand of the multiplication is 'alpha' (scalar),
  # and the left is the tensor.
  term = bin_op.left
  alpha = bin_op.right

  # 5. Construct new arguments
  # arg0: left_arg (preserve)
  # arg1: term (unwrapped from math operation)
  # kwarg: alpha=alpha

  # We ensure a comma exists after arg0, as arg1 follows
  arg0 = left_arg.with_changes(comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")))

  # We ensure a comma exists after arg1, as alpha kwarg follows
  arg1 = cst.Arg(value=term, comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")))

  new_args = [arg0, arg1]

  # Remove comma from alpha arg since it's last
  alpha_arg = cst.Arg(value=alpha, keyword=cst.Name("alpha"))

  new_args.append(alpha_arg)

  return node.with_changes(func=new_func, args=new_args)
