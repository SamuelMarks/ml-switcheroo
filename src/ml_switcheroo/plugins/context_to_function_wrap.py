"""
Plugin for handling Context Manager rewriting.

This module addresses the impedance mismatch between PyTorch's global state context managers
(like `torch.no_grad()`) and JAX's functional, explicit gradient handling.

It provides a transformation hook that:
1.  Detects usage of context managers flagged with `context_to_function_wrap`.
2.  Injects a `nullcontext` shim into the function preamble.
3.  Rewrites the specific API call to use this shim, ensuring the `with ...:` block
    remains valid Python syntax while effectively disabling gradient tracking semantics.
"""

import libcst as cst
from ml_switcheroo.core.hooks import register_hook, HookContext


def _create_dotted_name(name_str: str) -> cst.BaseExpression:
  """
  Creates a CST attribute chain from a string string.

  Args:
      name_str: The dotted path (e.g., 'contextlib.nullcontext').

  Returns:
      A LibCST Name or Attribute node.
  """
  parts = name_str.split(".")
  node = cst.Name(parts[0])
  for part in parts[1:]:
    node = cst.Attribute(value=node, attr=cst.Name(part))
  return node


@register_hook("context_to_function_wrap")
def transform_context_manager(node: cst.Call, ctx: HookContext) -> cst.Call:
  """
  Plugin Hook: Transforms valid Source context managers into JAX-compatible shims.

  Triggers:
      Operations marked with `requires_plugin: "context_to_function_wrap"` in Semantic JSONs.
      Primarily targets `torch.no_grad` and `torch.enable_grad`.

  Args:
      node: The original CST Call node (e.g., `torch.no_grad()`).
      ctx: The HookContext providing injection capabilities.

  Returns:
      The transformed CST Call node pointing to the injected shim.
  """
  # 1. Inject Imports
  ctx.inject_preamble("import contextlib")

  # 2. Rewrite the Function Name
  # We replace 'torch.no_grad' with 'contextlib.nullcontext'
  # This effectively makes the contents of the with-block run normally
  # without PyTorch's global gradient state affecting JAX (which ignores it anyway).
  new_func = _create_dotted_name("contextlib.nullcontext")

  # 3. Clear Arguments
  # Ensure the target call is empty to avoid passing invalid args (e.g. no_grad doesn't take args usually,
  # but some custom contexts might). nullcontext takes optional 'enter_result', usually safe to leave empty.
  empty_args = []

  return node.with_changes(func=new_func, args=empty_args)
