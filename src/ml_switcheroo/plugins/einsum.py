"""
Plugin for normalizing Einsum calls.

Standardizes `einsum` arguments so the equation string is always the first argument.
This is required because JAX enforces `einsum(equation, *operands)`, whereas
PyTorch historically allowed `einsum(operand, operand, equation)`.

This plugin handles both the argument shuffling AND the API renaming.
"""

import libcst as cst
from ml_switcheroo.core.hooks import register_hook, HookContext


def _create_dotted_name(name_str: str) -> cst.BaseExpression:
  """Creates a CST attribute chain from a string string."""
  parts = name_str.split(".")
  node = cst.Name(parts[0])
  for part in parts[1:]:
    node = cst.Attribute(value=node, attr=cst.Name(part))
  return node


def _is_string(node: cst.CSTNode) -> bool:
  """Checks if a node is a string literal."""
  return isinstance(node, (cst.SimpleString, cst.ConcatenatedString))


@register_hook("einsum_normalizer")
def normalize_einsum(node: cst.Call, ctx: HookContext) -> cst.Call:
  """
  Plugin Hook: Rotates arguments to place the equation string first and renames function.

  Triggers:
      Operations mapping to `einsum` with `requires_plugin: "einsum_normalizer"`.

  Args:
      node: The original CST Call node.
      ctx: HookContext.

  Returns:
      The transformed CST Call node.
  """
  # 1. Determine Target API Name
  target_api = ctx.lookup_api("einsum")

  # If lookup fails, keep original name (Removal of Hardcoded Fallback)
  new_func = _create_dotted_name(target_api) if target_api else node.func

  # 2. Argument Analysis
  if not node.args:
    return node.with_changes(func=new_func)

  args = list(node.args)

  # Check if first arg is string (already normalized)
  if _is_string(args[0].value):
    return node.with_changes(func=new_func)

  # 3. Find the string argument index (Search for equation)
  eq_idx = -1
  for i, arg in enumerate(args):
    # Check positional args only? Keywords generally shouldn't be equation mixed in pos
    if not arg.keyword and _is_string(arg.value):
      eq_idx = i
      break

  if eq_idx == -1:
    # No static string found (maybe variable passed or already correct).
    # Return matched function name but args untouched.
    return node.with_changes(func=new_func)

  # 4. Rotate
  eq_arg = args.pop(eq_idx)

  # Ensure the equation arg has a comma as it moves to the front
  # (assuming there are other operands if it wasn't first)
  if args:
    eq_arg = eq_arg.with_changes(comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")))

    # Clean comma of the new last arg (which was previously followed by eq or others)
    # We strip trailing comma to be safe/canonical
    last = args[-1]
    args[-1] = last.with_changes(comma=cst.MaybeSentinel.DEFAULT)

  args.insert(0, eq_arg)

  return node.with_changes(args=args, func=new_func)
