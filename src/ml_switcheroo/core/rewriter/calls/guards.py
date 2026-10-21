"""Strict Mode Guards Injection."""

import typing

import libcst as cst
from pydantic import JsonValue

if typing.TYPE_CHECKING:
  from ml_switcheroo.core.rewriter.calls.strategy import StrategyRewriterProtocol
  from ml_switcheroo.core.rewriter.interface import RewriterHost


class _HookCtxProtocol(typing.Protocol):
  """Minimal hook context protocol for strict guard preamble injection."""

  @property
  def metadata(self) -> dict[str, JsonValue]:
    """Return hook context metadata dictionary.

    Returns:
        Hook context metadata dictionary.
    """
    ...

  def inject_preamble(self, code_str: str) -> None:
    """Inject preamble code into the module.

    Args:
        code_str: Preamble code string to inject.
    """
    ...


class _ContextWithHookProtocol(typing.Protocol):
  """Minimal rewriter context protocol exposing hook_context."""

  @property
  def hook_context(self) -> _HookCtxProtocol:
    """Return hook context.

    Returns:
        Active hook context.
    """
    ...


class GuardRewriterProtocol(typing.Protocol):
  """Minimal rewriter protocol required by apply_strict_guards."""

  @property
  def context(self) -> _ContextWithHookProtocol:
    """Return rewriter context.

    Returns:
        Active rewriter context.
    """
    ...


STRICT_RANK_HELPER = """
def _check_rank(x, rank):
    if hasattr(x, 'ndim'): n = x.ndim
    elif hasattr(x, 'shape'): n = len(x.shape)
    else: return x
    assert n == rank, f"Strict Guard: Expected rank {rank}, got {n}"
    return x
"""


def apply_strict_guards(
  rewriter: "RewriterHost | StrategyRewriterProtocol | GuardRewriterProtocol",
  norm_args: list[cst.Arg],
  details: typing.Mapping[str, JsonValue | typing.Hashable | typing.Sized],
  target_impl: typing.Mapping[str, JsonValue | typing.Hashable | typing.Sized],
) -> list[cst.Arg]:
  """Wrap arguments with rank assertion helper calls if required by strict mode.

  Args:
      rewriter: The CST rewriter instance containing context.
      norm_args: The normalized list of arguments.
      details: Details of the standard API.
      target_impl: The target framework's implementation details.

  Returns:
      List[cst.Arg]: A new list of arguments, potentially wrapped in assertion logic.

  """
  raw_std_args = details.get("std_args", [])
  std_args = raw_std_args if isinstance(raw_std_args, list) else []
  raw_target_args = target_impl.get("args", {})
  target_arg_map: dict[str, str | None] = (
    {str(k): (str(v) if v is not None else None) for k, v in raw_target_args.items()}
    if isinstance(raw_target_args, dict)
    else {}
  )

  guards_map: dict[str, int] = {}
  for item in std_args:
    if isinstance(item, dict):
      r = item.get("rank")
      name = item.get("name")
      if isinstance(name, str) and name and isinstance(r, (int, float, str)):
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
