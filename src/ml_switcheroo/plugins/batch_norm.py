"""
Plugin for Functionalizing Batch Normalization (State Unwrapping).

Addresses the semantic mismatch between:
1.  **PyTorch (In-Place)**: `y = bn(x)`. Updates `running_mean`/`var` attributes on `bn` silently.
2.  **JAX/Flax (Functional)**: `y, new_state = bn(x, mutable=['batch_stats'])`. State updates are returned explicitly.

This plugin:
1.  Injects specific kwargs required by Flax (`use_running_average`, `mutable`).
2.  Adapts the return value to fit into an expression context by selecting the output tensor `[0]`.

**Limitation**:
This plugin solves the *Forward Pass* compatibility. It effectively discards the updated state
(`new_state`), effectively turning the layer into inference-mode regarding state persistence,
unless the surrounding code is manually refactored to handle the tuple return.
This is a necessary compromise to allow `y = bn(x)` to compile in JAX without extensive
dataflow analysis of the entire training loop.
"""

import libcst as cst
from ml_switcheroo.core.hooks import register_hook, HookContext


@register_hook("batch_norm_unwrap")
def transform_batch_norm(node: cst.Call, ctx: HookContext) -> cst.CSTNode:
  """
  Hook: Wraps BatchNorm calls to handle functional state returns.

  Transformation:
      Input:  `self.bn1(x)`
      Output: `self.bn1(x, use_running_average=not training, mutable=['batch_stats'])[0]`

  Logic:
      1.  **Mode Switching**: Injects `use_running_average=not training`.
          This assumes a `training` boolean variable exists in the scope (commonly injected
          by `src/ml_switcheroo/plugins/state_flag_injection.py`).
      2.  **Mutability**: Injects `mutable=['batch_stats']` to allow tracking stats during training.
      3.  **Unwrapping**: Applies `[0]` subscript to the result call. Flax returns `(tensor, updates)`,
          we select `tensor` to maintain compatibility with operators expecting a single array (like `relu`).

  Args:
      node: The original CST Call node.
      ctx: Hook Context containing target framework metadata.

  Returns:
      A CST Subscript node representing the tensor output of the BN call.
  """
  # 0. Context Check
  if ctx.target_fw not in ["jax", "flax", "flax_nnx"]:
    return node

  # 1. Prepare Arguments
  new_args = list(node.args)

  # 1a. Inject 'use_running_average'
  # Pattern: use_running_average = not training
  # Check if user manually supplied it first
  if not any(a.keyword and a.keyword.value == "use_running_average" for a in new_args):
    # Ensure previous arg has comma
    if new_args and new_args[-1].comma == cst.MaybeSentinel.DEFAULT:
      new_args[-1] = new_args[-1].with_changes(comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")))

    not_training_expr = cst.UnaryOperation(operator=cst.Not(), expression=cst.Name("training"))

    new_args.append(
      cst.Arg(
        keyword=cst.Name("use_running_average"),
        value=not_training_expr,
        equal=cst.AssignEqual(whitespace_before=cst.SimpleWhitespace(""), whitespace_after=cst.SimpleWhitespace(" ")),
      )
    )

  # 1b. Inject 'mutable'
  # Pattern: mutable=['batch_stats']
  if not any(a.keyword and a.keyword.value == "mutable" for a in new_args):
    # Ensure previous arg has comma
    if new_args and new_args[-1].comma == cst.MaybeSentinel.DEFAULT:
      new_args[-1] = new_args[-1].with_changes(comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")))

    batch_stats_list = cst.List(elements=[cst.Element(value=cst.SimpleString("'batch_stats'"))])

    new_args.append(
      cst.Arg(
        keyword=cst.Name("mutable"),
        value=batch_stats_list,
        equal=cst.AssignEqual(whitespace_before=cst.SimpleWhitespace(""), whitespace_after=cst.SimpleWhitespace(" ")),
      )
    )

  # 2. Reconstruct Call
  stateful_call = node.with_changes(args=new_args)

  # 3. Wrap in Subscript [0]
  # We select the first element (the output tensor)
  # The second element (the collection updates) is discarded in this AST replacement
  unwrapped_expression = cst.Subscript(
    value=stateful_call, slice=[cst.SubscriptElement(slice=cst.Index(value=cst.Integer("0")))]
  )

  return unwrapped_expression
