"""
Plugin for converting Flax NNX Variable definitions to PyTorch Parameters.

This module provides AST transformations to handle the impedance mismatch between
Flax NNX's explicit variable declarations (`nnx.Param`, `nnx.BatchStat`) and
PyTorch's `nn.Parameter` pattern.

It handles:
1.  **Trainable Parameters**: `nnx.Param(val)` -> `torch.nn.Parameter(val)`.
2.  **Non-Trainable State**: `nnx.BatchStat(val)` -> `torch.nn.Parameter(val, requires_grad=False)`.
    *Note: While PyTorch typically uses `register_buffer` for this, converting an Assignment
    expression to a `register_buffer` statement is structurally complex in AST replacement.
    Using non-grad Parameters is a semantic equivalent for state persistence.*
"""

import libcst as cst
from typing import Optional, List

from ml_switcheroo.core.hooks import register_hook, HookContext


def _create_dotted_name(name_str: str) -> cst.BaseExpression:
  """
  Creates a CST attribute chain from a string string.

  Args:
      name_str (str): The dotted path (e.g., 'torch.nn.Parameter').

  Returns:
      cst.BaseExpression: A LibCST Name or Attribute node.
  """
  parts = name_str.split(".")
  node = cst.Name(parts[0])
  for part in parts[1:]:
    node = cst.Attribute(value=node, attr=cst.Name(part))
  return node


@register_hook("nnx_param_to_torch")
def transform_nnx_param(node: cst.Call, ctx: HookContext) -> cst.Call:
  """
  Plugin Hook: Transforms valid NNX Variable declarations into PyTorch Parameters.

  Triggers:
      Operations marked with `requires_plugin: "nnx_param_to_torch"` in Semantic JSONs.
      Targeting: `flax.nnx.Param`, `flax.nnx.Variable`, `flax.nnx.BatchStat`.

  Transformation Logic:
      - Checks the source function signature/name via context or node analysis.
      - Maps `nnx.Param` -> `torch.nn.Parameter` (Trainable).
      - Maps `nnx.BatchStat`/`Variable` -> `torch.nn.Parameter(..., requires_grad=False)`.

  Args:
      node (cst.Call): The original CST Call node (e.g., `nnx.Param(zeros(1))`).
      ctx (HookContext): The HookContext containing config and semantics.

  Returns:
      cst.Call: The transformed CST Call node.
  """
  # 0. Safety Check: Only apply if targeting PyTorch
  if ctx.target_fw != "torch":
    return node

  # 1. Determine Source Type (Trainable vs Non-Trainable)
  # We infer this from the function name being replaced.
  # Note: The Rewriter calls us *before* renaming the function itself if the plugin
  # is handling the rewrite, or we check the original node structure.
  # We assume usage like `nnx.Param` or `nnx.BatchStat`.
  original_func_name = _extract_leaf_name(node.func)
  is_trainable = True

  if original_func_name in ["BatchStat", "Variable", "Cache", "Intermediate"]:
    is_trainable = False

  # 2. Construct New Function Name: torch.nn.Parameter
  new_func = _create_dotted_name("torch.nn.Parameter")

  # 3. Construct Arguments
  new_args: List[cst.Arg] = list(node.args)

  if not is_trainable:
    # Inject requires_grad=False
    req_grad_arg = cst.Arg(
      keyword=cst.Name("requires_grad"),
      value=cst.Name("False"),
      equal=cst.AssignEqual(
        whitespace_before=cst.SimpleWhitespace(""),
        whitespace_after=cst.SimpleWhitespace(""),
      ),
    )
    new_args.append(req_grad_arg)

  # 4. Return Transformed Node
  return node.with_changes(func=new_func, args=new_args)


def _extract_leaf_name(node: cst.BaseExpression) -> Optional[str]:
  """
  Helper to extract the right-most name from a call signature.

  Args:
      node (cst.BaseExpression): The function node (Name or Attribute).

  Returns:
      Optional[str]: The leaf identifier (e.g., 'Param' from 'nnx.Param') or None.
  """
  if isinstance(node, cst.Name):
    return node.value
  elif isinstance(node, cst.Attribute):
    return node.attr.value
  else:
    return None
