"""
Plugin for MultiHead Attention Strategy Selection.

This module provides modular hooks for transforming MultiHeadAttention API calls.
Unlike a monolithic function, it splits logic into discrete strategies (`repack_attn_keras`
and `repack_attn_flax`).

The main entry point `repack_attention_dispatch` uses the Abstract Operation definition
to select the appropriate strategy defined in the framework's JSON mapping.

Strategies:
1.  **Keras**: `repack_attn_keras`
    - Maps constructor: `embed_dim` -> `key_dim`.
    - Maps call: `(q, k, v, mask)` -> `(q, v, key=k, attention_mask=mask)`.

2.  **Flax/JAX**: `repack_attn_flax`
    - Maps constructor: Standard rename.
    - Maps call: `key_padding_mask` -> `mask`.
"""

import libcst as cst
from ml_switcheroo.core.hooks import register_hook, HookContext, get_hook


def _create_dotted_name(name_str: str) -> cst.BaseExpression:
  """Creates a CST attribute chain from a dotted string."""
  parts = name_str.split(".")
  node = cst.Name(parts[0])
  for part in parts[1:]:
    node = cst.Attribute(value=node, attr=cst.Name(part))
  return node


@register_hook("repack_attention_dispatch")
def repack_attention_dispatch(node: cst.Call, ctx: HookContext) -> cst.Call:
  """
  Dispatch Entry Point.
  Reads the `requires_plugin` field from the Semantics to determine the concrete strategy.

  If the variant specifies "repack_attention_dispatch" (itself), it attempts to
  guess based on target framework or fall through. However, modern adapters
  should point directly to `repack_attn_keras` or `repack_attn_flax`.
  """
  # Check if the variant specifies a specific sub-strategy
  # e.g. "requires_plugin": "repack_attn_keras"
  # But if this hook was called, it means "requires_plugin" was likely "repack_attention_dispatch"
  # or the rewriter resolved an alias.

  # We allow adapters to point to specific strategies directly in JSON.
  # If they point here, we dispatch based on target framework name (Legacy Backcompat).

  target_fw = ctx.target_fw.lower()

  if "keras" in target_fw:
    return repack_attn_keras(node, ctx)
  elif "flax" in target_fw or "jax" in target_fw:
    return repack_attn_flax(node, ctx)

  return node


@register_hook("repack_attn_keras")
def repack_attn_keras(node: cst.Call, ctx: HookContext) -> cst.Call:
  """
  Strategy: Keras Attention Packing.

  Constructor:
      torch: MultiheadAttention(embed_dim, num_heads)
      keras: MultiHeadAttention(num_heads, key_dim)

  Call:
      torch: forward(q, k, v, mask)
      keras: call(q, v, key=k, attention_mask=mask)
  """
  args = list(node.args)

  # --- Constructor Detection ---
  # Keras specific renaming logic
  if _is_constructor_signature(args):
    new_func = _create_dotted_name("keras.layers.MultiHeadAttention")
    new_args = []
    for arg in args:
      if arg.keyword:
        k = arg.keyword.value
        if k == "embed_dim":
          new_args.append(arg.with_changes(keyword=cst.Name("key_dim")))
        else:
          new_args.append(arg)
      else:
        # Blind positional preservation (usually safe here)
        new_args.append(arg)
    return node.with_changes(func=new_func, args=new_args)

  # --- Call Detection ---
  if len(args) >= 3:
    q_arg = args[0]
    k_arg = args[1]
    v_arg = args[2]
    remaining_args = args[3:]

    new_args = []
    # Keras expects (query, value, key=...)
    new_args.append(q_arg)

    # Ensure comma formatting for the new 2nd arg (value)
    v_arg_clean = v_arg.with_changes(comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")))
    new_args.append(v_arg_clean)

    # Convert Key to kwarg
    if k_arg:
      k_val = k_arg.value
      k_kw = cst.Arg(
        keyword=cst.Name("key"),
        value=k_val,
        equal=cst.AssignEqual(whitespace_before=cst.SimpleWhitespace(""), whitespace_after=cst.SimpleWhitespace("")),
        comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")),
      )
      # ensure comma on previous (v_arg_clean handled it)
      new_args.append(k_kw)

    # Map rest
    for arg in remaining_args:
      if arg.keyword:
        k = arg.keyword.value
        val = arg.value
        if k in ["attn_mask", "key_padding_mask"]:
          new_arg = arg.with_changes(keyword=cst.Name("attention_mask"), value=val)
          new_args.append(new_arg)
        else:
          new_args.append(arg)
      else:
        new_args.append(arg)

    return node.with_changes(args=new_args)

  return node


@register_hook("repack_attn_flax")
def repack_attn_flax(node: cst.Call, ctx: HookContext) -> cst.Call:
  """
  Strategy: Flax Attention Packing.

  Constructor:
      Standardizes to `flax.nnx.MultiHeadAttention`.

  Call:
      Maps `key_padding_mask` -> `mask`.
  """
  args = list(node.args)

  # Constructor
  if _is_constructor_signature(args):
    new_func = _create_dotted_name("flax.nnx.MultiHeadAttention")
    return node.with_changes(func=new_func)

  # Call
  if len(args) >= 3:
    q_arg = args[0]
    k_arg = args[1]
    v_arg = args[2]
    remaining_args = args[3:]

    new_args = [q_arg, k_arg, v_arg]

    for arg in remaining_args:
      if arg.keyword and arg.keyword.value in ["attn_mask", "key_padding_mask"]:
        new_args.append(arg.with_changes(keyword=cst.Name("mask")))
      else:
        new_args.append(arg)

    return node.with_changes(args=new_args)

  return node


def _is_constructor_signature(args: list) -> bool:
  """Heuristic to detect initialization vs forward call."""
  for arg in args:
    if arg.keyword and arg.keyword.value in ["embed_dim", "num_heads", "key_dim"]:
      return True

  # Fallback: very short positional args usually implies constructor
  # (embed_dim, num_heads) vs (q, k, v)
  if len(args) == 2 and not any(a.keyword for a in args):
    # Check if values are Integers?
    # If literals, safer guess.
    if isinstance(args[0].value, cst.Integer):
      return True

  return False
