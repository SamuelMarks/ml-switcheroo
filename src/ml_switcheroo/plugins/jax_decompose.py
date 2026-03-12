"""
JAX Fallback Decomposition Plugin for ml-switcheroo.

This plugin handles operations that have no direct mapping in the target framework.
It routes the AST call to the equivalent JAX operation, invokes the generic transpiler
on that JAX subgraph, and stitches the resulting AST back into the target.
"""

import libcst as cst
from typing import Union

from ml_switcheroo.core.hooks import register_hook, HookContext
from ml_switcheroo.plugins.utils import create_dotted_name


@register_hook("jax_decompose")
def decompose_via_jax(node: cst.Call, ctx: HookContext) -> Union[cst.Call, cst.BaseExpression]:
  """
  Translates an unsupported API call by routing it through JAX AST semantics.

  This acts as a fallback for complex math ops (e.g. `Hardswish`) when migrating to
  frameworks that lack them. Since JAX acts as our universal mathematical IR, we
  re-write the AST as if it were JAX, then optionally rely on subsequent passes
  to lower it further if needed.

  Args:
      node: The original CST Call node that has no direct mapping.
      ctx: Hook context containing target framework information.

  Returns:
      A rewritten CST node pointing to the JAX equivalent, or a custom lowered graph.
  """
  # 1. Fetch the abstract operation name from context
  op_name = ctx.current_op_id if ctx.current_op_id else "UnknownOp"

  # 2. In a real-world decomposition, we would:
  #    a) Query `operations.yaml` for the 'jax' variant of `op_name`.
  #    b) Inject the JAX dotted name or macro string.
  #    c) If the macro string exists, parse it into CST and substitute `{args}`.

  # For now, we simulate this by injecting a generic `jax.numpy` fallback call.
  # We prefix it to make it obvious it was decomposed.

  # We create: jax.numpy.<op_name>
  # Note: If the actual JAX API is known (from universal_mapping.json), we'd use that.
  new_func = create_dotted_name(f"jax.numpy.{op_name.lower()}")

  # 3. Preserve the arguments (they would be normalized by the semantic layer before reaching here)
  new_call = node.with_changes(func=new_func)

  # 4. We could technically transpile this `new_call` again into the target if
  #    we had an recursive AST transpiler exposed in HookContext, but for the plugin
  #    it's enough to map it to the intermediate representation.

  return new_call
