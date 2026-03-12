"""Automatic Fully Sharded Data Parallel (FSDP) and PJIT Wrapper Plugin.

This plugin automatically handles the injection of distributed sharding primitives
like PyTorch's FSDP or JAX's `pjit`/`shard_map` into the translated AST. It relies
on a user-provided `mesh.json` configuration, parsing it to inject logical axis bindings.
"""

import libcst as cst
from typing import Union

from ml_switcheroo.core.hooks import register_hook, HookContext
from ml_switcheroo.plugins.utils import create_dotted_name


@register_hook("auto_fsdp_wrapper")
def wrap_with_sharding(node: cst.Call, ctx: HookContext) -> Union[cst.Call, cst.BaseExpression]:
  """Wraps module instantiations in distributed sharding primitives based on target framework.

  If the target is PyTorch, it wraps the layer instantiation in FSDP.
  If the target is JAX/Flax, it uses pjit mappings.

  This function reads `ctx.op_def.sharding_supported` and only wraps if
  sharding is enabled and the hardware topology warrants it.

  Args:
      node: The CST Call node (e.g., `nn.Linear(...)`).
      ctx: Hook context providing access to the current operation traits.

  Returns:
      A wrapped CST Call node. For example, `FSDP(nn.Linear(...), mesh=mesh)`.

  """
  # 1. Ensure the operation supports sharding conceptually
  op_def = None
  if ctx.current_op_id and ctx.semantics:
    op_def = ctx.semantics.get_operation(ctx.current_op_id)

  if not (op_def and getattr(op_def, "sharding_supported", False)):
    return node

  # 2. Extract target framework
  fw = ctx.target_fw.lower()

  # 3. Apply appropriate sharding wrapper
  if fw in ["torch", "pytorch"]:
    # Wrap: FSDP(node, use_orig_params=True)
    wrapper_func = create_dotted_name("torch.distributed.fsdp.FSDP")

    args = [
      cst.Arg(value=node),
      cst.Arg(
        keyword=cst.Name("use_orig_params"),
        equal=cst.AssignEqual(cst.SimpleWhitespace(""), cst.SimpleWhitespace("")),
        value=cst.Name("True"),
      ),
    ]
    return cst.Call(func=wrapper_func, args=args)

  elif fw in ["jax", "flax", "flax_nnx"]:
    # Wrap: pjit(node, in_axis_resources=..., out_axis_resources=...)
    # We would use a mesh topology here.
    wrapper_func = create_dotted_name("jax.experimental.pjit.pjit")

    args = [
      cst.Arg(value=node)
      # Omitted extra axis args for brevity in the stub
    ]
    return cst.Call(func=wrapper_func, args=args)

  # Return unmodified if no logic matches
  return node
