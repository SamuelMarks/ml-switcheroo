"""
Plugin for converting Flax NNX Variable definitions to PyTorch-style Parameters.

This module provides AST transformations to handle the impedance mismatch between
Flax NNX's explicit variable declarations (`nnx.Param`, `nnx.BatchStat`) and
PyTorch's `nn.Parameter` pattern.

It handles:

1.  **Trainable Parameters**: `nnx.Param(val)` -> `TargetParam(val)`.
2.  **Non-Trainable State**: `nnx.BatchStat(val)` -> `TargetParam(val, requires_grad=False)`.

Decoupling:
    Logic is triggered solely by the `requires_plugin="nnx_param_to_torch"` wiring.
    The target class name is resolved via `ctx.lookup_api` based on the abstract Operation ID.
    If no mapping is found in the Knowledge Base, the transformation aborts (returns original node),
    preventing hardcoded fallbacks to `torch.nn.Parameter`.
"""

import libcst as cst
from typing import Optional, List

from ml_switcheroo.core.hooks import register_hook, HookContext


def _create_dotted_name(name_str: str) -> cst.BaseExpression:
  """
  Creates a CST attribute chain from a string string.
  """
  parts = name_str.split(".")
  node = cst.Name(parts[0])
  for part in parts[1:]:
    node = cst.Attribute(value=node, attr=cst.Name(part))
  return node


def _extract_leaf_name(node: cst.BaseExpression) -> Optional[str]:
  """
  Helper to extract the right-most name from a call signature.
  """
  if isinstance(node, cst.Name):
    return node.value
  elif isinstance(node, cst.Attribute):
    return node.attr.value
  else:
    return None


@register_hook("nnx_param_to_torch")
def transform_nnx_param(node: cst.Call, ctx: HookContext) -> cst.Call:
  """
  Plugin Hook: Transforms valid NNX Variable declarations into PyTorch-style Parameters.

  Triggers:
      Operations marked with `requires_plugin: "nnx_param_to_torch"`.
      Targeting: `flax.nnx.Param`, `flax.nnx.Variable`, `flax.nnx.BatchStat`.

  Logic:
      1. Determines if the source variable was trainable (`Param`) or not (`BatchStat`, `Variable`).
      2. Looks up the target API for the current operation.
      3. Injects `requires_grad=False` if not trainable.

  Args:
      node: The original CST Call node (e.g., `nnx.Param(zeros(1))`).
      ctx: The HookContext containing configuration.

  Returns:
      cst.Call: The transformed CST Call node or original if mapping missing.
  """
  # 1. Determine Source Type (Trainable vs Non-Trainable)
  # We infer this from the function name being replaced.
  original_func_name = _extract_leaf_name(node.func)
  is_trainable = True

  if original_func_name in ["BatchStat", "Variable", "Cache", "Intermediate"]:
    is_trainable = False

  # 2. Resolve Target API
  # Use context to look up what "Param" or the specific Op maps to in the target framework.
  # If lookup fails, we return the original node to avoid hallucinating APIs.
  target_api = ctx.lookup_api(ctx.current_op_id or "Param")
  if not target_api:
    return node

  new_func = _create_dotted_name(target_api)

  # 3. Construct Arguments
  new_args: List[cst.Arg] = list(node.args)

  # Only inject requires_grad if it's non-trainable
  if not is_trainable:
    # Check if 'requires_grad' is already present to avoid duplication
    if not any(a.keyword and a.keyword.value == "requires_grad" for a in new_args):
      # Inject requires_grad=False
      req_grad_arg = cst.Arg(
        keyword=cst.Name("requires_grad"),
        value=cst.Name("False"),
        equal=cst.AssignEqual(
          whitespace_before=cst.SimpleWhitespace(""),
          whitespace_after=cst.SimpleWhitespace(""),
        ),
      )

      # Ensure comma on previous arg
      if new_args:
        last = new_args[-1]
        if last.comma == cst.MaybeSentinel.DEFAULT:
          new_args[-1] = last.with_changes(comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")))

      new_args.append(req_grad_arg)

  # 4. Return Transformed Node
  return node.with_changes(func=new_func, args=new_args)
