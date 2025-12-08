"""
Plugin for translating Device Availability Checks.

Maps `torch.cuda.is_available()` to JAX's `len(jax.devices('gpu')) > 0`.
"""

import libcst as cst
from ml_switcheroo.core.hooks import register_hook, HookContext


@register_hook("cuda_is_available")
def transform_cuda_check(node: cst.Call, ctx: HookContext) -> cst.BaseExpression:
  """
  Plugin Hook: Transforms CUDA availability check.

  Triggers:
      `torch.cuda.is_available()` via 'cuda_is_available' plugin key.

  Transformation:
      Input:  `torch.cuda.is_available()`
      Output: `len(jax.devices('gpu')) > 0`

  Note: usage of 'gpu' backend string is hardcoded as per standard JAX idiom for CUDA.
  """
  if ctx.target_fw != "jax":
    return node

  # 1. Construct: jax.devices('gpu')
  # We use a direct construction to ensure 'gpu' string is used
  devices_call = cst.Call(
    func=cst.Attribute(value=cst.Name("jax"), attr=cst.Name("devices")),
    args=[cst.Arg(value=cst.SimpleString("'gpu'"))],
  )

  # 2. Construct: len(jax.devices('gpu'))
  len_call = cst.Call(func=cst.Name("len"), args=[cst.Arg(value=devices_call)])

  # 3. Construct: ... > 0
  # Comparison wraps the expression
  comparison = cst.Comparison(
    left=len_call,
    comparisons=[
      cst.ComparisonTarget(
        operator=cst.GreaterThan(),
        comparator=cst.Integer("0"),
      )
    ],
  )

  return comparison
