"""
Plugin for Type Casting Methods.

Addresses the syntax mismatch between:
1. PyTorch Shorthands: `x.float()`, `x.long()`, `x.half()`, etc.
2. JAX/NumPy/Array API: `x.astype(dtype)`.

Transformation:
1. Detects calls to known shorthand methods.
2. Changes the method name to `astype`.
3. Injects the corresponding JAX/Numpy dtype object as the argument.

Mappings:
    .float()  -> .astype(jnp.float32)
    .double() -> .astype(jnp.float64)
    .half()   -> .astype(jnp.float16)
    .long()   -> .astype(jnp.int64)
    .int()    -> .astype(jnp.int32)
    .bool()   -> .astype(jnp.bool_)
    .byte()   -> .astype(jnp.uint8)
"""

import libcst as cst
from ml_switcheroo.core.hooks import register_hook, HookContext


def _create_dotted_name(name_str: str) -> cst.BaseExpression:
  """Helper: Creates a CST Attribute chain from string."""
  parts = name_str.split(".")
  node = cst.Name(parts[0])
  for part in parts[1:]:
    node = cst.Attribute(value=node, attr=cst.Name(part))
  return node


# Static mapping table
TYPE_MAP = {
  "float": "jax.numpy.float32",
  "double": "jax.numpy.float64",
  "half": "jax.numpy.float16",
  "long": "jax.numpy.int64",
  "int": "jax.numpy.int32",
  "short": "jax.numpy.int16",
  "bool": "jax.numpy.bool_",
  "byte": "jax.numpy.uint8",
  "char": "jax.numpy.int8",
}


@register_hook("type_methods")
def transform_casting(node: cst.Call, ctx: HookContext) -> cst.Call:
  """
  Hook: Converts shorthand casts to astype calls.

  Trigger: Operations mapped to 'Cast' category or specific methods with `requires_plugin: "type_methods"`.
  """
  if ctx.target_fw not in ["jax", "numpy", "tensorflow"]:
    return node

  # We expect a method call: x.float()
  if not isinstance(node.func, cst.Attribute):
    return node

  method_name = node.func.attr.value

  # Check if we have a mapping for this method name
  if method_name not in TYPE_MAP:
    return node

  target_dtype_str = TYPE_MAP[method_name]

  # Construct new method name 'astype'
  new_func = node.func.with_changes(attr=cst.Name("astype"))

  # Construct dtype argument
  # If the user passed args (e.g. memory_format), we usually ignore them mostly for casts,
  # or append invalidly. Astype takes (dtype, ...).
  # We replace args with the dtype.

  dtype_node = _create_dotted_name(target_dtype_str)

  new_args = [cst.Arg(value=dtype_node)]

  return node.with_changes(func=new_func, args=new_args)
