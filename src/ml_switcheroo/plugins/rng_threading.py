"""
Plugin for RNG State Threading (The "JAX Pointer" Pattern).

PyTorch handles randomness via global state (`torch.manual_seed` and `generator` objects),
whereas JAX requires explicit passing and splitting of PRNG keys.

This plugin automates the transition by:
1.  **Signature Injection**: Adds an `rng` argument to the function definition.
2.  **Preamble Injection**: Adds `rng, key = jax.random.split(rng)` at the start of the function.
3.  **Call Rewriting**:
    - Appends `key=key` to supported stochastic calls.
    - Strips Torch-specific `generator` arguments (incompatible with JAX).

This ensures standard PyTorch calls like `torch.randn(..., generator=g)` become
`jax.random.normal(..., key=key)`.
"""

import libcst as cst
from typing import List
from ml_switcheroo.core.hooks import register_hook, HookContext


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

  Args:
      node: The original CST Call node (e.g., `torch.dropout(x, 0.5)`).
      ctx: The HookContext used to request global scope changes (signature/preamble).

  Returns:
      The transformed CST Call node with the 'key' keyword argument appended
      and 'generator' arguments removed.
  """
  # 0. Safety Check: Only apply JAX patterns if JAX is the target.
  # While the engine usually filters this, plugins can be defensive.
  if ctx.target_fw != "jax":
    return node

  # 1. Request Signature Injection
  # We ask the context to ensure 'rng' is present in the function definition.
  # The Rewrite Engine is responsible for checking if 'rng' already exists
  # to avoid duplication.
  ctx.inject_signature_arg("rng")

  # 2. Request Preamble Injection
  # We inject the splitting logic at the top of the function body.
  # This creates a fresh 'key' for use in the current scope.
  # The Rewrite Engine ensures this line is only added once per function.
  ctx.inject_preamble("rng, key = jax.random.split(rng)")

  # 3. Clean Arguments
  # Remove Torch-specific legacy RNG arguments like 'generator'
  cleaned_args = _remove_generator_arg(list(node.args))

  # 4. Modify the Call
  # Create the `key=key` argument node.
  key_arg = cst.Arg(
    keyword=cst.Name("key"),  # FIXED: was 'kword', correct is 'keyword'
    value=cst.Name("key"),
    equal=cst.AssignEqual(whitespace_before=cst.SimpleWhitespace(""), whitespace_after=cst.SimpleWhitespace("")),
  )

  # Append our new key argument to the cleaned list
  new_args = cleaned_args + [key_arg]

  # Return the modified node.
  # Note: The API name mapping (e.g., swapping torch.dropout -> jax.random.bernoulli)
  # is handled by the standard semantic mapping logic within the Rewriter
  # either before or after this hook runs, preserving the structure we build here.
  return node.with_changes(args=new_args)
