"""
Plugin for Type Casting Methods.

Addresses the syntax mismatch between:
1. PyTorch Shorthands: `x.float()`, `x.long()`, `x.half()`, etc.
2. JAX/NumPy/Array API: `x.astype(dtype)`.

Transformation:
1. Detects calls to known shorthand methods (triaged by Rewriter via 'type_methods' plugin).
2. Looks up the `target_type` in the semantic metadata for the abstract operation (e.g., `CastFloat` -> `Float32`).
3. Queries the Semantics Manager for the target framework's implementation of that Type (e.g., `Float32` -> `jnp.float32`).
4. Generates an `.astype(...)` call using the retrieved dtype API.
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


@register_hook("type_methods")
def transform_casting(node: cst.Call, ctx: HookContext) -> cst.Call:
  """
  Hook: Converts shorthand casts to astype calls.

  Logic:
      1. Access the Abstract Operation ID (e.g., 'CastFloat').
      2. Look up the 'target_type' metadata (e.g., 'Float32').
      3. Look up the target framework's API for 'Float32' (e.g., 'jax.numpy.float32').
      4. Rewrite `x.foo()` -> `x.astype(jax.numpy.float32)`.

  Args:
      node (cst.Call): The source call node.
      ctx (HookContext): Context providing access to the Knowledge Base.

  Returns:
      cst.Call: The transformed node utilizing `.astype()`.
  """
  # 0. Safety Checks
  # Only applicable if target framework uses numpy-like astype syntax
  if ctx.target_fw not in ["jax", "numpy", "tensorflow", "mlx", "flax", "flax_nnx"]:
    return node

  # We expect a method call: x.float()
  if not isinstance(node.func, cst.Attribute):
    return node

  # 1. Identify Target Abstract Type
  # The PivotRewriter has identified the operation (e.g. 'CastFloat') and set it in context
  op_id = ctx.current_op_id
  if not op_id:
    return node

  # Retrieve definition from Semantics Manager
  # This dictionary comes from standards_internal.py (or overrides)
  defn = ctx.semantics.get_definition_by_id(op_id)
  if not defn:
    return node

  # Extract Metadata: "metadata": {"target_type": "Float32"}
  target_type_id = defn.get("metadata", {}).get("target_type")

  # Fallback: Infer type from Op ID if metadata missing (e.g. CastFloat -> Float32)
  if not target_type_id:
    if op_id.startswith("Cast"):
      target_type_id = op_id[4:]
      # Mapping CastFloat -> Float32, CastLong -> Int64 requires map
      # Basic inference only handles direct suffixes
      if target_type_id == "Float":
        target_type_id = "Float32"
      if target_type_id == "Long":
        target_type_id = "Int64"
      if target_type_id == "Half":
        target_type_id = "Float16"
      if target_type_id == "Int":
        target_type_id = "Int32"
      if target_type_id == "Short":
        target_type_id = "Int16"
      if target_type_id == "Bool":
        target_type_id = "Bool"
      if target_type_id == "Byte":
        target_type_id = "UInt8"

  if not target_type_id:
    # If metadata is missing, we cannot resolve the type.
    return node

  # 2. Resolve Target Dtype API
  # Ask the semantics manager: "How does target_fw implement 'Float32'?"
  target_dtype_api = ctx.lookup_api(target_type_id)

  if not target_dtype_api:
    # If the target framework hasn't defined this type, we can't cast to it.
    return node

  # 3. Construct Transformation
  # Change method name to 'astype'
  new_func = node.func.with_changes(attr=cst.Name("astype"))

  # Create dtype argument node
  dtype_node = _create_dotted_name(target_dtype_api)

  # Replace arguments
  # Note: Shorthand casts like .float() in Torch take arguments (memory_format),
  # but .astype() signature is (dtype, ...). We strictly inject the dtype as the first arg.
  # Any existing arguments are usually incompatible or defaults, so we overwrite them.
  # If maintaining args is critical (e.g. copy=False), logic gets complex,
  # but standard ml-switcheroo policy handles the primary path first.
  new_args = [cst.Arg(value=dtype_node)]

  return node.with_changes(func=new_func, args=new_args)
