"""
Plugin for TopK Output Adaptation.

Addresses semantic mismatch:
1.  **PyTorch**: `values, indices = torch.topk(x, k)` (Returns named-tuple-like result).
2.  **JAX**: `values, indices = jax.lax.top_k(x, k)` (Returns tuple).

Transformation:
Wraps the target function call in a `collections.namedtuple` factory construction to maintain
attribute access (e.g. `.values`, `.indices`) while using a backend that returns raw tuples.

Decoupling Logic:
- Strict API lookup for "TopK".
- If not found, returns original node.
"""

import libcst as cst
from ml_switcheroo.core.hooks import register_hook, HookContext


def _create_dotted_name(name_str: str) -> cst.BaseExpression:
  """
  Creates a CST node structure representing a dotted path.

  Args:
      name_str: The dotted string (e.g. 'collections.namedtuple').

  Returns:
      A LibCST node (Name or nested Attribute).
  """
  parts = name_str.split(".")
  node = cst.Name(parts[0])
  for part in parts[1:]:
    node = cst.Attribute(value=node, attr=cst.Name(part))
  return node


@register_hook("topk_adapter")
def transform_topk(node: cst.Call, ctx: HookContext) -> cst.CSTNode:
  """
  Hook: Wraps target top_k call in a NamedTuple constructor.

  Orchestrates the following:

  1. Looks up the target API for "TopK" (e.g., `jax.lax.top_k`).
  2. Strips arguments not supported by the target (e.g., `largest`, `sorted`).
  3. Injects `import collections` into the file preamble.
  4. Wraps the call execution in a `collections.namedtuple` factory to restore
     `.values` and `.indices` accessors expected by Torch code.

  Args:
      node: The original CST Call node.
      ctx: HookContext for API lookup and preamble injection.

  Returns:
      cst.Call: The transformed call.
  """
  # 1. Resolve Target Function (Strict)
  target_api = ctx.lookup_api("TopK")
  if not target_api:
    return node

  inner_func = _create_dotted_name(target_api)

  # 2. Clean Arguments (Strip unsupported kwargs)
  clean_args = []
  args = list(node.args)

  for arg in args:
    kw = arg.keyword.value if arg.keyword else None
    if kw in ["largest", "sorted", "out"]:
      continue
    clean_arg = arg.with_changes(comma=cst.MaybeSentinel.DEFAULT)
    clean_args.append(clean_arg)

  inner_call = cst.Call(func=inner_func, args=clean_args)

  # 3. Create NamedTuple wrapper
  # We construct: collections.namedtuple("TopK", ["values", "indices"])(*inner_call)

  ctx.inject_preamble("import collections")

  factory = _create_dotted_name("collections.namedtuple")
  type_name = cst.SimpleString('"TopK"')
  fields_list = cst.List(
    elements=[
      cst.Element(value=cst.SimpleString('"values"')),
      cst.Element(
        value=cst.SimpleString('"indices"'),
        comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")),
      ),
    ]
  )
  factory_args = [
    cst.Arg(value=type_name, comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" "))),
    cst.Arg(value=fields_list),
  ]

  constructor_call = cst.Call(func=factory, args=factory_args)
  unpacked_arg = cst.Arg(value=inner_call, star="*")

  return cst.Call(func=constructor_call, args=[unpacked_arg])
