"""
Plugin for Type Casting Methods.

Addresses the syntax mismatch between:
1. PyTorch Shorthands: `x.float()`, `x.long()`, `x.half()`, etc.
2. JAX/NumPy/Array API: `x.astype(dtype)`.

Transformation:
1. Detects calls to known shorthand methods (triaged by Rewriter via 'type_methods' plugin).
2. Checks if the target framework declares `has_numpy_compatible_arrays` in its traits.
3. Looks up the `target_type` in the semantic metadata for the abstract operation.
4. Queries the Semantics Manager for the target framework's implementation of that Type.
5. Generates an `.astype(...)` call using the retrieved dtype API.
"""

import libcst as cst
from typing import Optional, Dict, Any

from ml_switcheroo.core.hooks import register_hook, HookContext


def _create_dotted_name(name_str: str) -> cst.BaseExpression:
  """Helper: Creates a CST Attribute chain from string."""
  parts = name_str.split(".")
  node = cst.Name(parts[0])
  for part in parts[1:]:
    node = cst.Attribute(value=node, attr=cst.Name(part))
  return node


def _supports_numpy_casting(ctx: HookContext) -> bool:
  """
  Checks if the target framework configuration supports numpy-style
  `.astype()` calling conventions via PluginTraits.
  """
  if not ctx.semantics:
    return False

  # Retrieve dict configuration for the active target framework
  conf: Dict[str, Any] = ctx.semantics.get_framework_config(ctx.target_fw)
  if not conf:
    return False

  # Navigate: config -> plugin_traits -> has_numpy_compatible_arrays
  # We handle both dict access (from JSON) and object access (if hydrated objects are used)
  traits = conf.get("plugin_traits")
  if not traits:
    return False

  if isinstance(traits, dict):
    return traits.get("has_numpy_compatible_arrays", False)

  if hasattr(traits, "has_numpy_compatible_arrays"):
    return traits.has_numpy_compatible_arrays

  return False


@register_hook("type_methods")
def transform_casting(node: cst.Call, ctx: HookContext) -> cst.Call:
  """
  Hook: Converts shorthand casts to astype calls.

  Logic:
      1. Verify target framework supports numpy array semantics (via Traits).
      2. Access the Abstract Operation ID (e.g., 'CastFloat').
      3. Look up the 'target_type' metadata (e.g., 'Float32').
      4. Look up the target framework's API for 'Float32' (e.g., 'jax.numpy.float32').
      5. Rewrite `x.foo()` -> `x.astype(jax.numpy.float32)`.

  Args:
      node (cst.Call): The source call node.
      ctx (HookContext): Context providing access to the Knowledge Base.

  Returns:
      cst.Call: The transformed node utilizing `.astype()`.
  """
  # 0. Capability Check (Decoupled from hardcoded strings)
  if not _supports_numpy_casting(ctx):
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
  # Useful for legacy or partial definitions.
  if not target_type_id:
    if op_id.startswith("Cast"):
      target_type_id = op_id[4:]
      # Mapping suffix to Abstract Type ID if different
      suffix_map = {
        "Float": "Float32",
        "Long": "Int64",
        "Half": "Float16",
        "Int": "Int32",
        "Short": "Int16",
        "Bool": "Bool",
        "Byte": "UInt8",
      }
      if target_type_id in suffix_map:
        target_type_id = suffix_map[target_type_id]

  if not target_type_id:
    # If metadata is missing/resolvable, we cannot safely transform.
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
  new_args = [cst.Arg(value=dtype_node)]

  return node.with_changes(func=new_func, args=new_args)
