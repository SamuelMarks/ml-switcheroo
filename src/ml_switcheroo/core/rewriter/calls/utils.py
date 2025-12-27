"""
Utility functions for Call Rewriting.

This module provides helper functions for inspecting and transforming LibCST Call nodes.
It handles structural tasks such as:

*   Detecting functional usage patterns (e.g. `layer.apply`).
*   Rewriting stateful calls.
*   Injecting and stripping keyword arguments generically.
*   Generating permutation/transpose calls based on semantic layout maps.

Decoupling Logic:
    Logic regarding specific framework APIs (e.g., whether to use `permute` vs `transpose`)
    is delegated to the `SemanticsManager`, removing hardcoded framework checks.
    Functional unwrapping detection is driven by `StructuralTraits`.
"""

from typing import Dict, Any, List, Optional, Tuple
import libcst as cst

from ml_switcheroo.utils.node_diff import diff_nodes
from ml_switcheroo.core.tracer import get_tracer
from ml_switcheroo.semantics.manager import SemanticsManager


def is_functional_apply(node: cst.Call, method_name: Optional[str] = "apply") -> bool:
  """
  Detects if a call node matches the functional execution pattern (e.g. `obj.apply`).

  Driven by the `functional_execution_method` trait of the source framework.
  This genericizes detection to support Flax (`apply`), Haiku (`apply`), or custom
  frameworks (`call_fn`).

  Args:
      node (cst.Call): The function call node to inspect.
      method_name (str, optional): The method name to look for. Defaults to "apply".
                                   If None, functional unwrapping is disabled.

  Returns:
      bool: True if the call is a method matching the name.
  """
  if not method_name:
    return False

  if isinstance(node.func, cst.Attribute):
    if node.func.attr.value == method_name:
      return True
  return False


def rewrite_stateful_call(rewriter: Any, node: cst.Call, instance_name: str, config: Dict[str, str]) -> cst.Call:
  """
  Rewrites a call to a stateful object to match a functional pattern.

  Used when converting OOP frameworks to Functional ones where state must be passed explicitly.
  Can inject arguments (e.g. `variables`) and change method names (e.g. `__call__` -> `apply`).

  Args:
      rewriter: The BaseRewriter instance (for access to signature stacks and context).
      node (cst.Call): The original call node.
      instance_name (str): The name of the object instance being called.
      config (Dict[str, str]): Configuration dict containing 'prepend_arg' and 'method'.

  Returns:
      cst.Call: The transformed call node.
  """
  new_args = list(node.args)
  target_arg_name = config.get("prepend_arg", "variables")

  # Track injection requirement in the enclosing function signature
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


def inject_kwarg(node: cst.Call, arg_name: str, val_name: str) -> cst.Call:
  """
  Generic helper to inject a keyword argument into a call.
  Prevents duplication if the argument already exists.

  Format: `func(..., arg_name=val_name)`

  Args:
      node (cst.Call): The call node to modify.
      arg_name (str): The keyword argument name.
      val_name (str): The variable name to pass as value.

  Returns:
      cst.Call: The updated call node with the injected argument (if not present).
  """
  # Duplicate check
  for arg in node.args:
    if arg.keyword and arg.keyword.value == arg_name:
      return node

  new_args = list(node.args)

  # Ensure comma formatting on the previous argument
  if len(new_args) > 0:
    last = new_args[-1]
    # Always force whitespace after the comma for style consistency
    new_args[-1] = last.with_changes(comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")))

  new_args.append(
    cst.Arg(
      keyword=cst.Name(arg_name),
      value=cst.Name(val_name),
      equal=cst.AssignEqual(
        whitespace_before=cst.SimpleWhitespace(""),
        whitespace_after=cst.SimpleWhitespace(""),
      ),
    )
  )
  return node.with_changes(args=new_args)


def inject_rngs_kwarg(node: cst.Call) -> cst.Call:
  """
  Legacy wrapper for `inject_kwarg`.
  Injects `rngs=rngs` into a call. Preserved for backward compatibility
  until all consumers migrate to `inject_kwarg`.

  Args:
      node (cst.Call): The call node.

  Returns:
      cst.Call: The updated call.
  """
  return inject_kwarg(node, "rngs", "rngs")


def strip_kwarg(node: cst.Call, kw_name: str) -> cst.Call:
  """
  Removes a specified keyword argument from a function call.

  Args:
      node (cst.Call): The call node.
      kw_name (str): The keyword string to strip.

  Returns:
      cst.Call: The updated node with the argument removed.
  """
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
  """
  Detects if a call is `super()` or `super().method()`.

  Args:
      node (cst.Call): The call node.

  Returns:
      bool: True if it is a super call.
  """
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
  """
  Checks if a name corresponds to a standard Python builtin.
  Used to prevent excessive logging/tracing of standard language features.

  Args:
      name (str): The function name.

  Returns:
      bool: True if builtin.
  """
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
    "list",
    "tuple",
    "dict",
    "set",
    "bool",
  }


def log_diff(label: str, original: cst.CSTNode, modified: cst.CSTNode) -> None:
  """
  Helper to compute AST diffs and log them to the tracer if changes occurred.

  Args:
      label (str): Label for the log entry.
      original (cst.CSTNode): The node before transformation.
      modified (cst.CSTNode): The node after transformation.
  """
  src_before, src_after, is_changed = diff_nodes(original, modified)
  if is_changed:
    get_tracer().log_mutation(label, src_before, src_after)


def compute_permutation(source_layout: str, target_layout: str) -> Optional[Tuple[int, ...]]:
  """
  Computes permutation indices to transform source layout string to target layout string.

  Example:
      Source: "NCHW", Target: "NHWC"
      Map: N:0, C:1, H:2, W:3
      Target Required: N(0), H(2), W(3), C(1)
      Result: (0, 2, 3, 1)

  Args:
      source_layout (str): Source layout string (e.g. "NCHW").
      target_layout (str): Target layout string (e.g. "NHWC").

  Returns:
      tuple[int, ...]: Tuple of integer indices, or None if invalid.
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
  Constructs a generic Call node from a name node and argument list.

  Args:
      func_name_node (cst.BaseExpression): The function identifier.
      args (List[cst.Arg]): List of arguments.

  Returns:
      cst.Call: The constructed call.
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

  Decoupling Logic:
      It queries the SemanticsManager for the `permute_dims` definition in the target tier.
      It does NOT contain hardcoded framework checks (like "if target == torch").
      If a definition is missing, it returns the bare node (No-Op), avoiding assumption
      of JAX-style syntax.

  Args:
      base_node (cst.CSTNode): The expression to wrap (the input tensor).
      indices (Tuple[int, ...]): Tuple of dimensions to permute (e.g., (0, 2, 3, 1)).
      semantics (SemanticsManager): Manager to look up syntax.
      target_fw (str): Target framework key.

  Returns:
      cst.CSTNode: Node representing `permute(base_node, indices)` or original if unsupported.
  """
  # 1. Lookup 'permute_dims' definition logic
  variant = semantics.resolve_variant("permute_dims", target_fw)

  # Strict failure: If target framework does not define how to permute,
  # we cannot generate a permute call safely. We return the original node.
  if not variant or not variant.get("api"):
    return base_node

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
      equal=cst.AssignEqual(
        whitespace_before=cst.SimpleWhitespace(""),
        whitespace_after=cst.SimpleWhitespace(""),
      ),
    )
    call_args.append(kw_arg)

  else:
    # Positional Varargs: .permute(x, 0, 2, 1)
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
