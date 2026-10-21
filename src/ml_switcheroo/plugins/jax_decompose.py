"""JAX Fallback Decomposition Plugin for ml-switcheroo.

This plugin handles operations that have no direct mapping in the target framework.
It routes the AST call to the equivalent JAX operation, invokes the generic transpiler
on that JAX subgraph, and stitches the resulting AST back into the target.
"""

import libcst as cst

from ml_switcheroo.core.hooks import HookContext, register_hook
from ml_switcheroo.plugins.utils import create_dotted_name


@register_hook("jax_decompose")
def decompose_via_jax(node: cst.Call, ctx: HookContext) -> cst.Call | cst.BaseExpression:
  """Translate an unsupported API call by routing it through JAX AST semantics.

  This acts as a fallback for complex math ops (e.g. `Hardswish`) when migrating to
  frameworks that lack them. Since JAX acts as our universal mathematical IR, we
  re-write the AST as if it were JAX, then optionally rely on subsequent passes
  to lower it further if needed.

  Args:
      node: The original CST Call node that has no direct mapping.
      ctx: Hook context containing target framework information.

  Returns:
      A rewritten CST node pointing to the target or JAX equivalent.

  """
  # 1. Fetch the abstract operation name from context
  op_name = ctx.current_op_id if ctx.current_op_id else "UnknownOp"

  # 2. Resolve target API from semantics if available and valid, otherwise fall back to jax.numpy.<op_name>
  resolved_api = ctx.lookup_api(op_name) if ctx.current_op_id else None
  target_api = (
    resolved_api
    if isinstance(resolved_api, str) and resolved_api.replace(".", "").isidentifier()
    else f"jax.numpy.{op_name.lower()}"
  )
  new_func = create_dotted_name(target_api)

  # 3. Preserve the arguments (they would be normalized by the semantic layer before reaching here)
  return node.with_changes(func=new_func)
