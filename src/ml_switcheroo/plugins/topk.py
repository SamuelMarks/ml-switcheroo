"""
Plugin for TopK Output Adaptation.
"""

import libcst as cst
from ml_switcheroo.core.hooks import register_hook, HookContext


def _create_dotted_name(name_str: str) -> cst.BaseExpression:
  parts = name_str.split(".")
  node = cst.Name(parts[0])
  for part in parts[1:]:
    node = cst.Attribute(value=node, attr=cst.Name(part))
  return node


@register_hook("topk_adapter")
def transform_topk(node: cst.Call, ctx: HookContext) -> cst.CSTNode:
  if ctx.target_fw not in ["jax", "flax", "flax_nnx"]:
    return node

  args = list(node.args)
  target_api = "jax.lax.top_k"
  inner_func = _create_dotted_name(target_api)

  clean_args = []
  for arg in args:
    kw = arg.keyword.value if arg.keyword else None
    if kw in ["largest", "sorted", "out"]:
      continue
    # Remove trailing comma to ensure inner call is clean
    clean_arg = arg.with_changes(comma=cst.MaybeSentinel.DEFAULT)
    clean_args.append(clean_arg)

  inner_call = cst.Call(func=inner_func, args=clean_args)

  factory = _create_dotted_name("collections.namedtuple")
  type_name = cst.SimpleString('"TopK"')
  fields_list = cst.List(
    elements=[
      cst.Element(value=cst.SimpleString('"values"')),
      cst.Element(value=cst.SimpleString('"indices"'), comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" "))),
    ]
  )
  factory_args = [
    cst.Arg(value=type_name, comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" "))),
    cst.Arg(value=fields_list),
  ]
  constructor = cst.Call(func=factory, args=factory_args)

  unpacked_arg = cst.Arg(value=inner_call, star="*")

  return cst.Call(func=constructor, args=[unpacked_arg])
