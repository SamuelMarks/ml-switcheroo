"""
Strict Mode Guards Injection.
"""

from typing import List, Dict, Any
import libcst as cst

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
  """Wraps args with rank assertions."""
  std_args = details.get("std_args", [])
  target_arg_map = target_impl.get("args", {})

  guards_map = {}
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
      found = [s for s, t in target_arg_map.items() if t == arg_key]
      if found:
        found_std_name = found[0]
      elif arg_key in guards_map:
        found_std_name = arg_key

    if found_std_name and found_std_name in guards_map:
      rank = guards_map[found_std_name]
      wrapper = cst.Call(
        func=cst.Name("_check_rank"),
        args=[
          cst.Arg(value=arg.value),
          cst.Arg(value=cst.Integer(str(rank))),
        ],
      )
      new_args.append(arg.with_changes(value=wrapper))
      guards_applied = True
    else:
      new_args.append(arg)

  if guards_applied:
    if hasattr(rewriter, "context") and not rewriter.context.hook_context.metadata.get("strict_helper_injected"):
      rewriter.context.hook_context.inject_preamble(STRICT_RANK_HELPER)
      rewriter.context.hook_context.metadata["strict_helper_injected"] = True

  return new_args
