"""Plugin for JIT Compilation and Device Barrier Ecosystem Mapping.

Rewrites JIT compilation decorators/wrappers and global device synchronization
calls strictly from ``HookContext.current_variant`` without hardcoded target
framework strings.
"""

import typing

import libcst as cst

from ml_switcheroo.core.hooks import HookContext, register_hook
from ml_switcheroo.plugins.utils import create_dotted_name, resolve_variant_api

_CALLABLE_KEYWORDS = frozenset({"fn", "fun", "func", "model"})


def _create_dotted_name(name_str: str) -> cst.BaseExpression:
  """Create a CST attribute chain from a dotted API path.

  Args:
      name_str: Dot-separated API path string.

  Returns:
      LibCST expression node corresponding to ``name_str``.
  """
  return create_dotted_name(name_str)


def _extract_compiled_target_arg(args: typing.Sequence[cst.Arg]) -> list[cst.Arg]:
  """Extract the primary callable/model argument from positional or keyword args.

  Args:
      args: Sequence of call arguments passed to ``compile(...)``.

  Returns:
      Single-element list containing the cleaned positional callable argument, or empty list.
  """
  for arg in args:
    if arg.keyword is None and not arg.star:
      return [arg.with_changes(comma=cst.MaybeSentinel.DEFAULT)]
  for arg in args:
    if arg.keyword is not None and arg.keyword.value in _CALLABLE_KEYWORDS:
      return [
        arg.with_changes(
          keyword=None,
          equal=cst.MaybeSentinel.DEFAULT,
          comma=cst.MaybeSentinel.DEFAULT,
        )
      ]
  return []


@register_hook("mlx_compiler")
def transform_compiler(node: cst.Decorator | cst.Call | cst.CSTNode, ctx: HookContext) -> cst.CSTNode:
  """Map JIT compilation decorators and functional wrappers via ``current_variant``.

  Args:
      node: Decorator or Call CST node representing the compilation target.
      ctx: Hook context providing variant and semantic API resolution.

  Returns:
      Transformed CST node, or ``node`` unchanged if ``Compile`` is unmapped.
  """
  target_api = resolve_variant_api(ctx, ("Compile", "compile"))
  if not target_api:
    return node

  new_func = _create_dotted_name(target_api)

  if isinstance(node, cst.Decorator):
    return node.with_changes(decorator=new_func)

  if isinstance(node, cst.Call):
    return node.with_changes(func=new_func, args=_extract_compiled_target_arg(node.args))

  return node


@register_hook("mlx_synchronize")
def transform_synchronize(node: cst.Call, ctx: HookContext) -> cst.CSTNode:
  """Map global barrier synchronization calls to an explicit diagnostic warning.

  Args:
      node: Synchronization CST Call node.
      ctx: Hook execution context.

  Returns:
      Print call warning that target lazy frameworks require explicit tensor arguments.
  """
  message_str = "# [Switcheroo] Global sync requires explicit tensor args in target framework."
  return cst.Call(func=cst.Name("print"), args=[cst.Arg(value=cst.SimpleString(f"'{message_str}'"))])
