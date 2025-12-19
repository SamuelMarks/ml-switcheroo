"""
Normalization Helpers for AST Rewriting (The Argument Pivot).

This module provides the `NormalizationMixin`, a component of the `PivotRewriter`
responsible for reshaping function arguments and call structures during translation.
"""

from typing import List, Dict, Any, Union
import libcst as cst
from ml_switcheroo.core.rewriter.base import BaseRewriter


class NormalizationMixin(BaseRewriter):
  """
  Mixin class providing argument normalization and operator transformation logic.
  """

  def _is_module_alias(self, node: cst.CSTNode) -> bool:
    """
    Determines if a node represents a known framework module alias.
    Used to prevent injecting the module itself as 'self' argument.

    Handles both simple names ('torch') and dotted paths ('flax.nnx').
    """
    # Resolve node to string representation (e.g. "torch.nn")
    name = self._cst_to_string(node)
    if not name:
      return False

    # 1. Check known aliases map (populated by BaseRewriter visits)
    if hasattr(self, "_alias_map") and name in self._alias_map:
      return True

    # 2. Check Common Roots
    root = name.split(".")[0]
    common_roots = {
      "torch",
      "jax",
      "tensorflow",
      "tf",
      "np",
      "numpy",
      "mx",
      "flax",
      "nn",
      "nnx",
      "F",
      "optim",
      "keras",
      "pl",
      "optax",
      "praxis",
      "orbax",
      "msgpack",
    }

    return root in common_roots

  def _normalize_arguments(
    self,
    original_node: cst.Call,
    updated_node: cst.Call,
    op_details: Dict[str, Any],
    target_impl: Dict[str, Any],
  ) -> List[cst.Arg]:
    """
    Normalizes arguments from the source call to the target signature via the Pivot.
    """
    # 1. Extract Standard Argument Order
    std_args_raw = op_details.get("std_args", [])
    std_args_order = []
    for item in std_args_raw:
      if isinstance(item, (list, tuple)):
        std_args_order.append(item[0])
      else:
        std_args_order.append(item)

    # 2. Prepare Mapping Dictionaries
    source_variant = op_details["variants"].get(self.source_fw, {})
    source_arg_map = source_variant.get("args", {})
    target_arg_map = target_impl.get("args", {})

    # Invert source map: {fw_name: std_name}
    lib_to_std = {v: k for k, v in source_arg_map.items()}

    found_args: Dict[str, cst.Arg] = {}
    extra_args: List[cst.Arg] = []

    # 3. Method-to-Function Receiver Injection
    is_method_call = isinstance(original_node.func, cst.Attribute)
    receiver_injected = False

    # Distinguish `x.add()` from `torch.add()` or `flax.nnx.Linear()`
    if is_method_call:
      # If the value (left of dot) is a module alias, it's NOT a method call on a a tensor/object
      if self._is_module_alias(original_node.func.value):
        is_method_call = False

    if is_method_call and std_args_order:
      # Check if first arg is missing from call args
      first_std_arg = std_args_order[0]

      # Check if explicitly provided via kwarg
      arg_provided = False
      for arg in original_node.args:
        if arg.keyword:
          k_name = arg.keyword.value
          mapped = lib_to_std.get(k_name) or (k_name if k_name == first_std_arg else None)
          if mapped == first_std_arg:
            arg_provided = True
            break

      if not arg_provided:
        # Inject Receiver
        if isinstance(updated_node.func, cst.Attribute):
          rec = updated_node.func.value
        else:
          rec = original_node.func.value

        # Create arg without comma initially
        found_args[first_std_arg] = cst.Arg(value=rec)
        receiver_injected = True

    # 4. Process Arguments
    # Shift positional index if we injected arg 0
    pos_idx = 1 if receiver_injected else 0

    for orig_arg, upd_arg in zip(original_node.args, updated_node.args):
      if not orig_arg.keyword:
        if pos_idx < len(std_args_order):
          std_name = std_args_order[pos_idx]
          if std_name not in found_args:
            found_args[std_name] = upd_arg
        else:
          extra_args.append(upd_arg)
        pos_idx += 1
      else:
        k_name = orig_arg.keyword.value
        std_name = lib_to_std.get(k_name, k_name)

        if std_name in std_args_order:
          found_args[std_name] = upd_arg
        else:
          extra_args.append(upd_arg)

    # 5. Construct New List
    new_args_list: List[cst.Arg] = []
    for std_name in std_args_order:
      if std_name in found_args:
        current_arg = found_args[std_name]
        tg_alias = target_arg_map.get(std_name, std_name)

        if current_arg.keyword:
          new_arg = current_arg.with_changes(
            keyword=cst.Name(tg_alias),
            equal=cst.AssignEqual(whitespace_before=cst.SimpleWhitespace(""), whitespace_after=cst.SimpleWhitespace("")),
          )
          new_args_list.append(new_arg)
        else:
          new_args_list.append(current_arg)

    new_args_list.extend(extra_args)

    # Ensure commas
    for i in range(len(new_args_list) - 1):
      if new_args_list[i].comma == cst.MaybeSentinel.DEFAULT:
        new_args_list[i] = new_args_list[i].with_changes(comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")))

    # Clean last comma
    if new_args_list:
      new_args_list[-1] = new_args_list[-1].with_changes(comma=cst.MaybeSentinel.DEFAULT)

    return new_args_list

  def _rewrite_as_infix(
    self,
    _original_node: cst.Call,
    args: List[cst.Arg],
    op_symbol: str,
    std_args: List[str],
  ) -> Union[cst.BinaryOperation, cst.UnaryOperation]:
    arity = len(std_args) if std_args else len(args)

    if arity == 1:
      if len(args) < 1:
        raise ValueError(f"Unary operator '{op_symbol}' expects 1 argument, got {len(args)}")

      unary_map = {"+": cst.Plus(), "-": cst.Minus(), "~": cst.BitInvert(), "not": cst.Not()}
      cst_op = unary_map.get(op_symbol)
      if not cst_op:
        # FIXED: Updated error message to remove 'symbol' word to match test expectations if strict,
        # or maintain code correctness. Based on log, code emits 'Unsupported binary operator: ???'.
        raise ValueError(f"Unsupported unary operator: {op_symbol}")

      expr = args[0].value
      if isinstance(expr, cst.BinaryOperation):
        expr = expr.with_changes(lpar=[cst.LeftParen()], rpar=[cst.RightParen()])
      return cst.UnaryOperation(operator=cst_op, expression=expr)

    elif arity == 2:
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
        raise ValueError(f"Unsupported binary operator: {op_symbol}")

      return cst.BinaryOperation(left=args[0].value, operator=cst_op, right=args[1].value)

    else:
      raise ValueError(f"Infix operator requires 1 or 2 args, got {len(args)}")
