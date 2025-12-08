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

**Configuration**:
Users can customize the variable naming via `pyproject.toml` or CLI args:
- `rng_arg_name`: Name of the argument injected into signature (default: "rng").
- `key_var_name`: Name of the local key variable split from rng (default: "key").
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
      ctx: The HookContext used to request global scope changes (signature/preamble)
           and read configuration.

  Returns:
      The transformed CST Call node with the 'key' keyword argument appended
      and 'generator' arguments removed.
  """
  # 0. Safety Check: Only apply JAX patterns if JAX is the target.
  # While the engine usually filters this, plugins can be defensive.
  if ctx.target_fw != "jax":
    return node

  # 1. Configuration
  # Read variable name preferences from user config
  rng_arg = ctx.raw_config("rng_arg_name", default="rng")
  key_var = ctx.raw_config("key_var_name", default="key")

  # 2. Request Signature Injection
  # We ask the context to ensure 'rng' (or custom name) is present in the function definition.
  # The Rewrite Engine is responsible for checking if it already exists to avoid duplication.
  ctx.inject_signature_arg(rng_arg)

  # 3. Request Preamble Injection
  # We inject the splitting logic at the top of the function body.
  # Example: "rng, key = jax.random.split(rng)"
  split_stmt = f"{rng_arg}, {key_var} = jax.random.split({rng_arg})"
  ctx.inject_preamble(split_stmt)

  # 4. Clean Arguments
  # Remove Torch-specific legacy RNG arguments like 'generator'
  cleaned_args = _remove_generator_arg(list(node.args))

  # 5. Modify the Call
  # Create the `key=key_var` argument node.
  # Note: The keyword "key" matches standard JAX API (e.g. random.normal(key=...)).
  # The value matches the local variable we split in the preamble.
  key_arg = cst.Arg(
    keyword=cst.Name("key"),
    value=cst.Name(key_var),
    equal=cst.AssignEqual(whitespace_before=cst.SimpleWhitespace(""), whitespace_after=cst.SimpleWhitespace("")),
  )

  # Append our new key argument to the cleaned list
  new_args = cleaned_args + [key_arg]

  return node.with_changes(args=new_args)
