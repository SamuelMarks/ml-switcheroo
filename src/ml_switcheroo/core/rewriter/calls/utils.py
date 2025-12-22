"""
Utility functions for Call Rewriting.

Validates function patterns and handles structural injections like state objects,
shims, and layout permutations.
"""

from typing import Dict, Any, List, Optional, Tuple
import libcst as cst
from ml_switcheroo.utils.node_diff import diff_nodes
from ml_switcheroo.core.tracer import get_tracer
from ml_switcheroo.semantics.manager import SemanticsManager


def is_functional_apply(node: cst.Call) -> bool:
  """
  Detects if a call node matches the `obj.apply` pattern used in Flax Linen.
  """
  if isinstance(node.func, cst.Attribute):
    if node.func.attr.value == "apply":
      return True
  return False


def rewrite_stateful_call(rewriter, node: cst.Call, instance_name: str, config: Dict[str, str]) -> cst.Call:
  """Rewrites a call to a stateful object (Functional patterns only)."""
  new_args = list(node.args)
  target_arg_name = config.get("prepend_arg", "variables")

  if rewriter._signature_stack:
    sig_ctx = rewriter._signature_stack[-1]
    if target_arg_name not in sig_ctx.existing_args:
      found = any(n == target_arg_name for n, _ in sig_ctx.injected_args)
      if not found:
        sig_ctx.injected_args.append((target_arg_name, None))
        rewriter._report_warning(f"Injected missing state argument '{target_arg_name}' into signature.")

  injected_arg = cst.Arg(
    value=cst.Name(target_arg_name),
    comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")),
  )
  new_args.insert(0, injected_arg)

  method_name = config.get("method")
  if method_name:
    new_func = cst.Attribute(
      value=rewriter._create_dotted_name(instance_name),
      attr=cst.Name(method_name),
    )
  else:
    new_func = node.func

  return node.with_changes(func=new_func, args=new_args)


def inject_rngs_kwarg(node: cst.Call) -> cst.Call:
  """Injects `rngs=rngs` into a constructor call."""
  # Duplicate check
  for arg in node.args:
    if arg.keyword and arg.keyword.value == "rngs":
      return node

  new_args = list(node.args)

  # Ensure comma on previous arg
  if len(new_args) > 0:
    last = new_args[-1]
    # Always force whitespace after the comma
    new_args[-1] = last.with_changes(comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")))

  new_args.append(
    cst.Arg(
      keyword=cst.Name("rngs"),
      value=cst.Name("rngs"),
      equal=cst.AssignEqual(
        whitespace_before=cst.SimpleWhitespace(""),
        whitespace_after=cst.SimpleWhitespace(""),
      ),
    )
  )
  return node.with_changes(args=new_args)


def strip_kwarg(node: cst.Call, kw_name: str) -> cst.Call:
  """Removes a keyword argument from a call node."""
  filtered = []
  for arg in node.args:
    if arg.keyword and arg.keyword.value == kw_name:
      continue
    filtered.append(arg)

  # Clean trailing comma on new last arg
  if filtered and filtered[-1].comma != cst.MaybeSentinel.DEFAULT:
    last = filtered[-1]
    filtered[-1] = last.with_changes(comma=cst.MaybeSentinel.DEFAULT)

  return node.with_changes(args=filtered)


def is_super_call(node: cst.Call) -> bool:
  """Helper to identify direct super() usage or super().__init__()."""
  if isinstance(node.func, cst.Attribute):
    # Case: super().method()
    receiver = node.func.value
    if isinstance(receiver, cst.Call) and isinstance(receiver.func, cst.Name):
      if receiver.func.value == "super":
        return True
  elif isinstance(node.func, cst.Name):
    # Case: super()
    if node.func.value == "super":
      return True
  return False


def is_builtin(name: str) -> bool:
  """Avoid spamming logs for standard python builtins unless mapped."""
  return name in {
    "print",
    "len",
    "range",
    "super",
    "enumerate",
    "zip",
    "int",
    "float",
    "str",
  }


def log_diff(label: str, original: cst.CSTNode, modified: cst.CSTNode) -> None:
  """Helper to compute diff and log if changed."""
  src_before, src_after, is_changed = diff_nodes(original, modified)
  if is_changed:
    get_tracer().log_mutation(label, src_before, src_after)


def compute_permutation(source_layout: str, target_layout: str) -> Optional[Tuple[int, ...]]:
  """
  Computes permutation indices to transform source layout to target.

  Example:
      Source: "NCHW", Target: "NHWC"
      Map: N:0, C:1, H:2, W:3
      Target Required: N(0), H(2), W(3), C(1)
      Result: (0, 2, 3, 1)

  Args:
      source_layout: Source layout string (e.g. "NCHW").
      target_layout: Target layout string (e.g. "NHWC").

  Returns:
      Tuple of integer indices, or None if invalid.
  """
  if len(source_layout) != len(target_layout):
    return None

  # Index map
  src_map = {char: i for i, char in enumerate(source_layout)}
  indices = []

  for char in target_layout:
    if char not in src_map:
      return None
    indices.append(src_map[char])

  return tuple(indices)


def inject_custom_api_call(
  func_name_node: cst.BaseExpression,
  args: List[cst.Arg],
) -> cst.Call:
  """
  Constructs a generic Call node.
  """
  return cst.Call(func=func_name_node, args=args)


def inject_permute_call(
  base_node: cst.CSTNode,
  indices: Tuple[int, ...],
  semantics: SemanticsManager,
  target_fw: str,
) -> cst.CSTNode:
  """
  Wraps a CST node with a permutation call valid for the target framework.

  Logic:
      1. Finds `permute_dims` definition in Semantics for the target framework.
      2. Determines API name (e.g. `jnp.transpose`, `torch.permute`, `tf.transpose`).
      3. Determines packing strategy (tuple kwarg vs varargs).
      4. Wraps `base_node`.

  Args:
      base_node: The expression to verify/permute.
      indices: Tuple of dimensions to permute (e.g., (0, 2, 3, 1)).
      semantics: Manager to look up `permute_dims` syntax.
      target_fw: Target framework key.

  Returns:
      CST Node representing `permute(base_node, indices)`.
  """
  # 1. Lookup 'permute_dims' definition
  # Note: permute_dims is standard in 'k_array_api.json' or 'internal'.
  # We assume it is available.
  variant = semantics.resolve_variant("permute_dims", target_fw)

  if not variant or not variant.get("api"):
    # Fallback if no definition found: Assume JAX syntax as safe default for engine
    # or handle error gracefully.
    # Let's create a synthesized variant for JAX/NumPy style as generic fallback.
    variant = {"api": "transpose", "pack_to_tuple": "axes"}
    if target_fw in ["torch"]:
      variant = {"api": "permute"}

  api_str = variant["api"]
  pack_kw = variant.get("pack_to_tuple")

  # 2. Build API Name Node
  parts = api_str.split(".")
  func_node = cst.Name(parts[0])
  for part in parts[1:]:
    func_node = cst.Attribute(value=func_node, attr=cst.Name(part))

  # 3. Construct Args
  # Input argument
  input_arg = cst.Arg(
    value=base_node,
    comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")),
  )

  # Indices arguments
  call_args = [input_arg]

  if pack_kw:
    # Keyword Tuple: .transpose(x, axes=(0, 2, 1))
    elements = []
    for idx_val in indices:
      elements.append(
        cst.Element(
          value=cst.Integer(str(idx_val)),
          comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")),
        )
      )
    # Clean last comma for cleanliness
    if elements:
      elements[-1] = elements[-1].with_changes(comma=cst.MaybeSentinel.DEFAULT)

    tuple_node = cst.Tuple(elements=elements)

    kw_arg = cst.Arg(
      keyword=cst.Name(pack_kw),
      value=tuple_node,
      equal=cst.AssignEqual(whitespace_before=cst.SimpleWhitespace(""), whitespace_after=cst.SimpleWhitespace("")),
    )
    call_args.append(kw_arg)

  else:
    # Positional Varargs: .permute(x, 0, 2, 1)
    # Note check if input arg needs comma
    # Iterate and add simple args
    for i, idx_val in enumerate(indices):
      comma = cst.Comma(whitespace_after=cst.SimpleWhitespace(" "))
      if i == len(indices) - 1:
        comma = cst.MaybeSentinel.DEFAULT

      call_args.append(
        cst.Arg(
          value=cst.Integer(str(idx_val)),
          comma=comma,
        )
      )

  return cst.Call(func=func_node, args=call_args)
