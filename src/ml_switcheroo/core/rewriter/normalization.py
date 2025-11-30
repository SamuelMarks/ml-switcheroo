"""
Normalization Helpers for AST Rewriting (The Argument Pivot).

This module provides the `NormalizationMixin`, a component of the `PivotRewriter`
responsible for reshaping function arguments and call structures during translation.

## The Pivot Strategy

The core challenge is translating `FrameworkA.op(a=1, b=2)` to `FrameworkB.op(x=1, y=2)`
where the argument names and order might differ. We solve this using an
"Abstract Standard" (The Pivot) derived from the Array API or ONNX specs.

1.  **Ingest (Source -> Abstract)**:
    Map provided source arguments to Abstract names.
    e.g., Source `dim` maps to Abstract `axis`.

2.  **Pivot (Alignment)**:
    Reorder arguments to match the Abstract Standard signature `(x, axis, keepdims)`.

3.  **Project (Abstract -> Target)**:
    Map Abstract names to Target framework argument names.
    e.g., Abstract `axis` maps to Target `axes`.

**Recursive Transformation Note**:
By using `updated_node` for values and `original_node` for structure, we ensure
that nested calls (e.g., `abs(neg(x))`) are correctly transformed inside-out
before being reordered.
"""

from typing import List, Dict, Any, Union
import libcst as cst
from ml_switcheroo.core.rewriter.base import BaseRewriter


class NormalizationMixin(BaseRewriter):
  """
  Mixin class providing argument normalization and operator transformation logic.

  Designed to be mixed into `PivotRewriter`. It relies on `self.semantics` to
  retrieve the argument specifications (`std_args`) and mapping rules.
  """

  def _normalize_arguments(
    self,
    original_node: cst.Call,
    updated_node: cst.Call,
    op_details: Dict[str, Any],
    target_impl: Dict[str, Any],
  ) -> List[cst.Arg]:
    """
    Normalizes arguments from the source call to the target signature via the Pivot.

    It correlates the *original* arguments (to identify keywords like 'dim') with
    the *updated* arguments (to preserve recursive transformations like 'torch.neg' -> 'jax.negative').

    Args:
        original_node: The LibCST `Call` node before child transformation.
        updated_node: The LibCST `Call` node after child transformation.
        op_details: The semantic definition containing `std_args` (The Abstract Spec).
        target_impl: The target variant definition containing argument mappings.

    Returns:
        A list of `cst.Arg` nodes formatted for the target function call.

    Raises:
        ValueError: If argument mapping is ambiguous or fails structural validation.
    """
    # 1. Extract Standard Argument Order (Names only)
    # std_args can be ["x", "axis"] or [("x", "Array"), ("axis", "int")]
    std_args_raw = op_details.get("std_args", [])
    std_args_order = []
    for item in std_args_raw:
      if isinstance(item, (list, tuple)):
        std_args_order.append(item[0])
      else:
        std_args_order.append(item)

    # 2. Prepare Mapping Dictionaries
    #   source_arg_map: {std_name: source_lib_name}
    #   target_arg_map: {std_name: target_lib_name}
    source_variant = op_details["variants"].get(self.source_fw, {})
    source_arg_map = source_variant.get("args", {})
    target_arg_map = target_impl.get("args", {})

    found_args: Dict[str, cst.Arg] = {}
    extra_args: List[cst.Arg] = []

    # 3. Process Arguments (Positional & Keyword)
    # We iterate `original` to check keywords/positions, but we store `updated`
    # to preserve transformations.
    pos_idx = 0

    # Safety: Zip assumes 1:1 mapping between original and updated children.
    # LibCST visitors guarantee this unless args were explicitly removed/added
    # by a child visitor, which should not happen in standard expression walking.
    for orig_arg, upd_arg in zip(original_node.args, updated_node.args):
      if not orig_arg.keyword:
        # --- Type A: Positional Argument ---
        if pos_idx < len(std_args_order):
          std_name = std_args_order[pos_idx]
          found_args[std_name] = upd_arg
        else:
          # Arguments exceeding the spec length (e.g., varargs)
          extra_args.append(upd_arg)
        pos_idx += 1
      else:
        # --- Type B: Keyword Argument ---
        k_name = orig_arg.keyword.value
        lib_to_std = {v: k for k, v in source_arg_map.items()}

        # Priority:
        # A. Explicit map (input -> x)
        # B. Identity fallback (x -> x) if matches spec
        std_name = lib_to_std.get(k_name)
        if not std_name and k_name in std_args_order:
          std_name = k_name

        if std_name:
          found_args[std_name] = upd_arg
        else:
          # Unknown keyword -> Passthrough (e.g. kwargs)
          extra_args.append(upd_arg)

    # 4. Construct New Argument List
    # Reassemble based on standard order using Target Names.
    # If standard arg 'x' was found, we emit it using the Target Name 'a'.
    new_args_list: List[cst.Arg] = []

    for std_name in std_args_order:
      if std_name in found_args:
        # Retrieve the UPDATED arg node
        current_arg = found_args[std_name]

        # Determine target keyword name
        # If target has specific mapping (e.g. axis->keepdims), use it.
        # Otherwise default to standard name.
        target_alias = target_arg_map.get(std_name, std_name)

        if current_arg.keyword:
          # If it was a keyword argument, rename the keyword
          new_arg = current_arg.with_changes(
            keyword=cst.Name(target_alias),
            equal=cst.AssignEqual(
              whitespace_before=cst.SimpleWhitespace(""),
              whitespace_after=cst.SimpleWhitespace(""),
            ),
          )
          new_args_list.append(new_arg)
        else:
          # Positional args remain positional to preserve code style
          new_args_list.append(current_arg)

    # Append any unmapped extras (varargs or unknown kwargs)
    new_args_list.extend(extra_args)
    return new_args_list

  def _rewrite_as_infix(
    self,
    _original_node: cst.Call,
    args: List[cst.Arg],
    op_symbol: str,
    std_args: List[str],
  ) -> Union[cst.BinaryOperation, cst.UnaryOperation]:
    """
    Transforms a standard Function Call into a Binary or Unary Operator expression.

    Useful for mapping API calls like `torch.add(a, b)` to `a + b` or
    `torch.neg(x)` to `-x`. Supports typical Python operators like
    `+`, `-`, `*`, `@` (matmul), `~` (invert), etc.

    Args:
      _original_node: The original CST Call (unused, kept for interface consistency).
      args: The normalized list of arguments (containing transformed values).
      op_symbol: The operator symbol string (e.g., "+", "-", "@", "not").
      std_args: The standard argument list from the spec (used to check arity).

    Returns:
      A CST `BinaryOperation` or `UnaryOperation` node representing the expression.

    Raises:
      ValueError: If the argument count doesn't match operator requirements (Unary=1, Binary=2).
    """
    # Determine arity (argument count)
    arity = len(std_args) if std_args else len(args)

    # --- Unary Operations ---
    if arity == 1:
      if len(args) != 1:
        raise ValueError(f"Unary operator '{op_symbol}' expects 1 argument, got {len(args)}")

      unary_map = {
        "+": cst.Plus(),
        "-": cst.Minus(),
        "~": cst.BitInvert(),
        "not": cst.Not(),
      }
      cst_op = unary_map.get(op_symbol)
      if not cst_op:
        raise ValueError(f"Unsupported unary operator symbol: {op_symbol}")

      # Wrap complex expressions in Parentheses to preserve precedence
      # Example: negation of an addition `neg(a + b)` -> `-(a + b)`
      expression = args[0].value
      if isinstance(expression, cst.BinaryOperation):
        expression = expression.with_changes(
          lpar=[cst.LeftParen()],
          rpar=[cst.RightParen()],
        )

      return cst.UnaryOperation(operator=cst_op, expression=expression)

    # --- Binary Operations ---
    elif arity == 2:
      # Ensure we have enough args regardless of std_args definition
      if len(args) < 2:
        raise ValueError(f"Binary operator '{op_symbol}' requires 2 arguments, got {len(args)}")

      op_map = {
        "+": cst.Add(),
        "-": cst.Subtract(),
        "*": cst.Multiply(),
        "/": cst.Divide(),
        "//": cst.FloorDivide(),
        "%": cst.Modulo(),
        "**": cst.Power(),
        "@": cst.MatrixMultiply(),
        "&": cst.BitAnd(),
        "|": cst.BitOr(),
        "^": cst.BitXor(),
        "<<": cst.LeftShift(),
        ">>": cst.RightShift(),
      }

      cst_op = op_map.get(op_symbol)
      if not cst_op:
        raise ValueError(f"Unsupported binary operator symbol: {op_symbol}")

      first = args[0].value
      second = args[1].value
      return cst.BinaryOperation(left=first, operator=cst_op, right=second)

    else:
      raise ValueError(f"Infix/Prefix operator requires 1 or 2 arguments, got {len(args)} (arity={arity})")
