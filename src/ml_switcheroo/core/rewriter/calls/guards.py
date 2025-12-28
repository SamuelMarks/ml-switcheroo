"""
Strict Mode Guards Injection.

This module helps inject runtime assertions for tensor properties (e.g., Rank)
when strict mode is enabled in the compiler configuration. It ensures that
transpiled code respects the constraints defined in the Semantic Specification.
"""

from typing import List, Dict, Any
import libcst as cst

# Helper Code for Preamble injection (Compact form)
STRICT_RANK_HELPER = """
def _check_rank(x, rank):
    if hasattr(x, 'ndim'): n = x.ndim
    elif hasattr(x, 'shape'): n = len(x.shape)
    else: return x
    assert n == rank, f"Strict Guard: Expected rank {rank}, got {n}"
    return x
"""


def apply_strict_guards(
  rewriter: Any,
  norm_args: List[cst.Arg],
  details: Dict[str, Any],
  target_impl: Dict[str, Any],
) -> List[cst.Arg]:
  """
  Injects runtime rank assertion wrappers around arguments if 'rank' constraint exists.

  If a standard argument has a `rank` constraint in the ODL spec, this function wraps
  the corresponding call argument with `_check_rank(val, N)`. It also handles injecting
  the helper function definition into the module preamble via the Rewriter context.

  Args:
      rewriter: The calling Rewriter instance (InvocationMixin).
      norm_args: The normalized list of CST Arguments.
      details: The Abstract Operation definition (Hub).
      target_impl: The Target Framework variant definition (Spoke).

  Returns:
      List[cst.Arg]: The list of arguments, potentially wrapped with guards.
  """
  std_args = details.get("std_args", [])
  target_arg_map = target_impl.get("args", {})

  # Identify Std Arguments that have constraints
  guards_map = {}  # {std_name: rank_int}

  for item in std_args:
    if isinstance(item, dict):
      r = item.get("rank")
      name = item.get("name")
      if name and r is not None:
        guards_map[name] = int(r)

  if not guards_map:
    return norm_args

  new_args = []
  guards_applied = False

  for arg in norm_args:
    arg_key = arg.keyword.value if arg.keyword else None
    found_std_name = None

    if arg_key:
      # Invert: target_arg_map (std -> tgt)
      # We look for which standard arg maps to this keyword
      found = [s for s, t in target_arg_map.items() if t == arg_key]
      if found:
        found_std_name = found[0]
      elif arg_key in guards_map:  # Identity mapping match
        found_std_name = arg_key
    else:
      # Positional usage requires positional index matching logic,
      # which is complex if reordering occurred.
      # For strict guards, we currently skip positional args unless
      # normalization guaranteed keywords.
      pass

    if found_std_name and found_std_name in guards_map:
      rank = guards_map[found_std_name]
      wrapper = cst.Call(
        func=cst.Name("_check_rank"),
        args=[
          cst.Arg(value=arg.value),
          cst.Arg(value=cst.Integer(str(rank))),
        ],
      )
      # Create new arg with the wrapped value, preserving keyword/comma
      new_args.append(arg.with_changes(value=wrapper))
      guards_applied = True
    else:
      new_args.append(arg)

  # Clean injection: Only inject preamble if we actually used a guard
  if guards_applied:
    if not rewriter.ctx.metadata.get("strict_helper_injected"):
      # Try to use BaseRewriter's module preamble list directly to avoid nesting issues
      if hasattr(rewriter, "_module_preamble"):
        if STRICT_RANK_HELPER not in rewriter._module_preamble:
          rewriter._module_preamble.append(STRICT_RANK_HELPER)
      else:
        # Fallback to context-based injection
        rewriter.ctx.inject_preamble(STRICT_RANK_HELPER)

      rewriter.ctx.metadata["strict_helper_injected"] = True

  return new_args
