"""
Plugin for RNG State Threading (The "JAX Pointer" Pattern).

PyTorch handles randomness via global state (`torch.manual_seed` and `generator` objects),
whereas JAX requires explicit passing and splitting of PRNG keys.

This plugin automates the transition by:
1.  **Signature Injection**: Adds an `rng` argument to the function definition.
2.  **Preamble Injection**: Asks the target Adapter for the correct split syntax
    (e.g. `rng, key = jax.random.split(rng)`) and injects it.
3.  **Call Rewriting**:
    - Appends `key=key` to supported stochastic calls.
    - Strips Torch-specific `generator` arguments (incompatible with JAX).

Decoupling:
Uses `traits.requires_explicit_rng` logic to determine execution, and
calls `adapter.get_rng_split_syntax()` to determine code generation.
"""

import libcst as cst
from typing import List

from ml_switcheroo.core.hooks import register_hook, HookContext
from ml_switcheroo.frameworks import get_adapter


def _remove_generator_arg(args: List[cst.Arg]) -> List[cst.Arg]:
  """
  Filters out the 'generator' keyword argument commonly used in PyTorch.
  JAX uses the 'key' semantics instead.
  """
  clean_args = []
  for arg in args:
    if arg.keyword and arg.keyword.value == "generator":
      continue
    clean_args.append(arg)
  return clean_args


@register_hook("inject_prng")
def inject_prng_threading(node: cst.Call, ctx: HookContext) -> cst.Call:
  """
  Plugin Hook: Thread PRNG keys for stochastic operations.

  Triggers:
      Operations marked with `requires_plugin: "inject_prng"` in the Semantic Knowledge Base.
      Examples: `torch.nn.functional.dropout`, `torch.randn`, `torch.bernoulli`.

  Logic:
      Checks `ctx.plugin_traits.requires_explicit_rng`. If True, applies JAX-style threading.
      This allows any framework (not just JAX) to opt-in to this behavior via configuration.

  Args:
      node: The original CST Call node (e.g., `torch.dropout(x, 0.5)`).
      ctx: The HookContext used to request global scope changes (signature/preamble)
           and read configuration.

  Returns:
      The transformed CST Call node with the 'key' keyword argument appended
      and 'generator' arguments removed.
  """
  # 0. Capability Check (Decoupled from Framework strings)
  if not ctx.plugin_traits.requires_explicit_rng:
    return node

  # 1. Configuration
  rng_arg = ctx.raw_config("rng_arg_name", default="rng")
  key_var = ctx.raw_config("key_var_name", default="key")

  # 2. Request Signature Injection
  ctx.inject_signature_arg(rng_arg)

  # 3. Request Preamble Injection (Delegated to Adapter)
  adapter = get_adapter(ctx.target_fw)
  if adapter:
    # Ask adapter for the syntax: "rng, key = jax.random.split(rng)"
    split_stmt = adapter.get_rng_split_syntax(rng_arg, key_var)
    if split_stmt and split_stmt != "pass":
      ctx.inject_preamble(split_stmt)

  # 4. Clean Arguments
  cleaned_args = _remove_generator_arg(list(node.args))

  # 5. Modify the Call
  key_arg = cst.Arg(
    keyword=cst.Name("key"),
    value=cst.Name(key_var),
    equal=cst.AssignEqual(
      whitespace_before=cst.SimpleWhitespace(""),
      whitespace_after=cst.SimpleWhitespace(""),
    ),
  )

  new_args = cleaned_args + [key_arg]

  return node.with_changes(args=new_args)
