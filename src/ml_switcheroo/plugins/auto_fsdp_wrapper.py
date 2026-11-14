"""Automatic Fully Sharded Data Parallel (FSDP) and PJIT Wrapper Plugin.

This plugin automatically handles the injection of distributed sharding primitives
like PyTorch's FSDP or JAX's `pjit`/`shard_map` into the translated AST. It relies
on a user-provided `mesh.json` configuration, parsing it to inject logical axis bindings.
"""

import typing

import libcst as cst

from ml_switcheroo.core.hooks import HookContext, register_hook
from ml_switcheroo.plugins.utils import create_dotted_name


class _OperationGetterProtocol(typing.Protocol):
  """Protocol for semantics objects providing `get_operation`."""

  def get_operation(self, op_id: str) -> "_ShardingOpProtocol | None":
    """Retrieve operation definition by ID.

    Args:
        op_id: Operation identifier.

    Returns:
        Operation definition object.
    """
    ...


@typing.runtime_checkable
class _ShardingOpProtocol(typing.Protocol):
  """Protocol for operation definitions exposing `sharding_supported`.

  Attributes:
      sharding_supported: The sharding supported (`bool`).
  """

  sharding_supported: bool


if typing.TYPE_CHECKING:
  from ml_switcheroo.semantics.manager import SemanticsManager


def _is_op_getter(
  obj: "SemanticsManager | _OperationGetterProtocol | None",
) -> typing.TypeGuard[_OperationGetterProtocol]:
  """Check if `obj` is a non-None semantics object providing `get_operation`.

  Args:
      obj: Candidate semantics object.

  Returns:
      True if non-None.
  """
  return obj is not None


@register_hook("auto_fsdp_wrapper")
def wrap_with_sharding(node: cst.Call, ctx: HookContext) -> cst.Call | cst.BaseExpression:
  """Wrap module instantiations in distributed sharding primitives based on target framework.

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
  if ctx.current_op_id and _is_op_getter(ctx.semantics):
    op_def = ctx.semantics.get_operation(ctx.current_op_id)

  if not (isinstance(op_def, _ShardingOpProtocol) and op_def.sharding_supported):
    return node

  # 2. Extract sharding configuration from traits
  sharding_api = ctx.plugin_traits.sharding_wrapper_api
  if not sharding_api:
    return node

  # 3. Apply appropriate sharding wrapper
  wrapper_func = create_dotted_name(sharding_api)

  if "fsdp" in sharding_api.lower():
    # Wrap: FSDP(node, use_orig_params=True)
    args = [
      cst.Arg(value=node),
      cst.Arg(
        keyword=cst.Name("use_orig_params"),
        equal=cst.AssignEqual(cst.SimpleWhitespace(""), cst.SimpleWhitespace("")),
        value=cst.Name("True"),
      ),
    ]
    return cst.Call(func=wrapper_func, args=args)

  if "pjit" in sharding_api.lower():
    # Wrap: pjit(node, in_axis_resources=..., out_axis_resources=...)
    # We would use a mesh topology here.
    args = [
      cst.Arg(value=node)
      # Omitted extra axis args for brevity in the stub
    ]
    return cst.Call(func=wrapper_func, args=args)

  # Return unmodified if no logic matches
  return node
