"""
Plugin for MultiHead Attention Strategy Selection.

This module provides modular hooks for transforming MultiHeadAttention API calls
between frameworks. Logic is split into discrete argument-mapping strategies
(`repack_attn_keras` and `repack_attn_flax`).

Decoupling Logic:
    - **No Hardcoded Frameworks:** The plugin does not contain strings like
      `flax.nnx` or `keras.layers`.
    - **Strict Lookup:** Target class names are resolved via `ctx.lookup_api`.
    - **Safety:** If the Knowledge Base is missing a mapping for "MultiheadAttention",
      constructor transformations are aborted to prevent hallucination.
"""

import libcst as cst
from typing import Optional
from ml_switcheroo.core.hooks import register_hook, HookContext


def _create_dotted_name(name_str: str) -> cst.BaseExpression:
  """Creates a CST attribute chain from a dotted string."""
  parts = name_str.split(".")
  node = cst.Name(parts[0])
  for part in parts[1:]:
    node = cst.Attribute(value=node, attr=cst.Name(part))
  return node


def _resolve_target_class(ctx: HookContext) -> Optional[cst.BaseExpression]:
  """
  Look up 'MultiheadAttention' implementation from the Knowledge Base.

  Returns:
      CST Node for the target class, or None if mapping is missing.
  """
  api = ctx.lookup_api("MultiheadAttention")
  if not api:
    return None
  return _create_dotted_name(api)


def _is_constructor_signature(args: list) -> bool:
  """Heuristic to detect initialization vs forward call."""
  for arg in args:
    if arg.keyword and arg.keyword.value in ["embed_dim", "num_heads", "key_dim"]:
      return True

  # Fallback: very short positional args usually implies constructor
  # (embed_dim, num_heads) vs (q, k, v)
  # Constructor usually takes ints
  if len(args) == 2 and not any(a.keyword for a in args):
    # Check if values are Integers
    if isinstance(args[0].value, cst.Integer):
      return True

  return False


@register_hook("repack_attn_keras")
def repack_attn_keras(node: cst.Call, ctx: HookContext) -> cst.Call:
  """
  Strategy: Keras Attention Packing.

  **Constructor:**
  - Requires 'MultiheadAttention' mapping in Semantics.
  - Renames `embed_dim` -> `key_dim` recursively.

  **Call (Inference):**
  - Remaps typical Torch signature `(q, k, v, mask)` to Keras `(q, v, key=k, attention_mask=mask)`.

  Args:
      node: Original Call node.
      ctx: HookContext for API lookup.

  Returns:
      Transformed Call node, or original if dependencies missing.
  """
  args = list(node.args)

  # --- Constructor Detection ---
  if _is_constructor_signature(args):
    # Strict Decoupling: Helper returns None if no mapping exists
    new_func = _resolve_target_class(ctx)
    if not new_func:
      return node

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
        equal=cst.AssignEqual(
          whitespace_before=cst.SimpleWhitespace(""),
          whitespace_after=cst.SimpleWhitespace(""),
        ),
        comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")),
      )
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
  Strategy: Flax/JAX Attention Packing.

  **Constructor:**
  - Requires 'MultiheadAttention' mapping in Semantics.

  **Call (Inference):**
  - Maps `key_padding_mask` -> `mask`.

  Args:
      node: Original Call node.
      ctx: HookContext for API lookup.

  Returns:
      Transformed Call node, or original if dependencies missing.
  """
  args = list(node.args)

  # Constructor
  if _is_constructor_signature(args):
    # Strict Decoupling: Helper returns None if no mapping exists
    new_func = _resolve_target_class(ctx)
    if not new_func:
      return node
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
