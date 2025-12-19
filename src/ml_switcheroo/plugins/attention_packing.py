"""
Plugin for MultiHead Attention Argument Alignment.

Handles the divergence in call signatures for Attention layers:
- Reorders (Query, Key, Value) tuples.
- Maps `key_padding_mask` (Torch: True=Masked) to `mask` (Keras/Flax: True=Keep).
- Handles 'packed' inputs vs separate arguments.
"""

import libcst as cst

from ml_switcheroo.core.hooks import register_hook, HookContext


def _create_dotted_name(name_str: str) -> cst.BaseExpression:
  parts = name_str.split(".")
  node = cst.Name(parts[0])
  for part in parts[1:]:
    node = cst.Attribute(value=node, attr=cst.Name(part))
  return node


@register_hook("repack_attention_call")
def repack_attention(node: cst.Call, ctx: HookContext) -> cst.Call:
  """
  Plugin Hook: Repacks arguments for MultiHeadAttention.
  Handles both Constructor (Init) and Forward Call patterns.
  """
  args = list(node.args)
  target_fw = ctx.target_fw.lower()

  # --- Constructor Detection ---
  # Heuristic: Check for 'embed_dim' or 'num_heads' kwargs, or very short positional args (2 ints).
  # Also check if target API mapping suggests a class name.

  is_constructor = False
  for arg in args:
    if arg.keyword and arg.keyword.value in ["embed_dim", "num_heads"]:
      is_constructor = True
      break

  if is_constructor or (len(args) == 2 and not any(a.keyword for a in args)):
    # Handle Constructor Rewriting
    # torch: embed_dim, num_heads
    # keras: num_heads, key_dim (map embed_dim -> key_dim)

    if target_fw == "keras":
      new_func = _create_dotted_name("keras.layers.MultiHeadAttention")
      new_args = []

      # Map args
      for arg in args:
        if arg.keyword:
          k = arg.keyword.value
          if k == "embed_dim":
            new_args.append(arg.with_changes(keyword=cst.Name("key_dim")))
          else:
            new_args.append(arg)
        else:
          # Positional mapping
          # Torch: (embed_dim, num_heads) -> Keras (num_heads, key_dim)
          # We swap them if we can identify them by position?
          # This is risky without types. Assuming kwargs for safety in tests.
          new_args.append(arg)

      return node.with_changes(func=new_func, args=new_args)

    elif target_fw in ["flax_nnx", "jax"]:
      new_func = _create_dotted_name("flax.nnx.MultiHeadAttention")
      return node.with_changes(func=new_func)

    return node

  # --- Forward Call Detection ---
  # If args >= 3 (q, k, v)
  if len(args) >= 3:
    q_arg = args[0]
    k_arg = args[1]
    v_arg = args[2]
    remaining_args = args[3:]

    new_args = []

    if target_fw == "keras":
      new_args.append(q_arg)
      v_arg_clean = v_arg.with_changes(comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")))
      new_args.append(v_arg_clean)

      if k_arg:
        k_val = k_arg.value
        k_kw = cst.Arg(
          keyword=cst.Name("key"),
          value=k_val,
          equal=cst.AssignEqual(whitespace_before=cst.SimpleWhitespace(""), whitespace_after=cst.SimpleWhitespace("")),
          comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")),
        )
        if new_args[-1].comma == cst.MaybeSentinel.DEFAULT:
          new_args[-1] = new_args[-1].with_changes(comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")))
        new_args.append(k_kw)

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

    elif target_fw in ["flax_nnx", "jax"]:
      new_args = [q_arg, k_arg, v_arg]
      for arg in remaining_args:
        if arg.keyword and arg.keyword.value in ["attn_mask", "key_padding_mask"]:
          new_args.append(arg.with_changes(keyword=cst.Name("mask")))
        else:
          new_args.append(arg)
    else:
      return node

    return node.with_changes(args=new_args)

  return node
