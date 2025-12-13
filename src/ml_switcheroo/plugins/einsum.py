"""
Plugin for normalizing Einsum calls.

Standardizes `einsum` arguments so the equation string is always the first argument.
JAX strictly enforces `einsum(equation, *operands)`, whereas other frameworks (like older
PyTorch versions or specific utility wrappers) might allow flexible ordering like
`einsum(operand, operand, equation)`.

This plugin handles:
1.  **Equation Identification**: Scans arguments to find the string literal (the equation).
2.  **Reordering**: Moves the equation to the 0th position if it isn't already there.
3.  **API Renaming**: Updates the function call to the target framework's API (e.g., `jax.numpy.einsum`).
4.  **Syntax Cleaning**: Essential comma management when shuffling argument order in the AST.
"""

import libcst as cst
from typing import List, Optional

from ml_switcheroo.core.hooks import register_hook, HookContext


def _create_dotted_name(name_str: str) -> cst.BaseExpression:
  """
  Creates a CST attribute chain from a dotted string string.

  Args:
      name_str: Dotted path (e.g. 'jax.numpy.einsum').

  Returns:
      CST node (Name or Attribute).
  """
  parts = name_str.split(".")
  node = cst.Name(parts[0])
  for part in parts[1:]:
    node = cst.Attribute(value=node, attr=cst.Name(part))
  return node


def _is_string(node: cst.CSTNode) -> bool:
  """Checks if a node is a string literal (Simple or Concatenated)."""
  return isinstance(node, (cst.SimpleString, cst.ConcatenatedString))


@register_hook("einsum_normalizer")
def normalize_einsum(node: cst.Call, ctx: HookContext) -> cst.Call:
  """
  Plugin Hook: Rotates arguments to place the equation string first and renames function.

  Triggers:
      Operations mapping to `Einsum` with `requires_plugin: "einsum_normalizer"`.

  Transformation:
      Input:  `torch.einsum(x, y, "ij,jk->ik")`
      Output: `jax.numpy.einsum("ij,jk->ik", x, y)`

  Args:
      node: The original CST Call node.
      ctx: HookContext for looking up the target API.

  Returns:
      The transformed CST Call node.
  """
  # 1. Determine Target API Name
  # We check standard casing 'Einsum' first, then lowercase 'einsum'
  target_api = ctx.lookup_api("Einsum") or ctx.lookup_api("einsum")

  # If lookup fails, we default to preserving the original name structure,
  # though this usually implies a configuration error in the semantics.
  new_func = _create_dotted_name(target_api) if target_api else node.func

  # 2. Argument Analysis
  if not node.args:
    # No arguments to normalize, just rename
    return node.with_changes(func=new_func)

  args: List[cst.Arg] = list(node.args)

  # Fast path: Check if first arg is already the equation (string literal)
  if _is_string(args[0].value):
    return node.with_changes(func=new_func)

  # 3. Scan for the equation argument
  eq_idx = -1
  for i, arg in enumerate(args):
    # We only look at positional args or keyword args if the key isn't explicit?
    # Einsum usually takes positional args.
    # We assume the equation is a string literal.
    if _is_string(arg.value):
      eq_idx = i
      break

  if eq_idx == -1:
    # No static string found (maybe variable passed or already correct/dynamic).
    # Return matched function name usage but leave args untouched.
    return node.with_changes(func=new_func)

  # 4. Rotation Logic
  # Extract the equation argument
  eq_arg = args.pop(eq_idx)

  # Prepare equation arg for 0th position (needs trailing comma + spacing)
  eq_arg = eq_arg.with_changes(comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")))

  # 5. Syntax Cleanup on remaining args
  # The argument that was previously at the end might have moved, or the one
  # before the equation might now be at the end.
  if args:
    # Ensure the current last argument doesn't have a trailing comma
    # (clean style, though Python allows it)
    last_arg = args[-1]
    if last_arg.comma != cst.MaybeSentinel.DEFAULT:
      args[-1] = last_arg.with_changes(comma=cst.MaybeSentinel.DEFAULT)

  # Insert equation at the front
  args.insert(0, eq_arg)

  return node.with_changes(args=args, func=new_func)
