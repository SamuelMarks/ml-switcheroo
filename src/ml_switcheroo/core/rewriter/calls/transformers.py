"""
AST Transformation Helpers.

Provides logic to reconstruct CST nodes for Infix operators, Inline Lambdas,
and Structured Index Selection.
"""

import libcst as cst
from typing import List, Union
from ml_switcheroo.utils.node_diff import capture_node_source


def apply_index_select(inner_node: cst.CSTNode, index: int) -> cst.Subscript:
  """
  Wraps an expression node with a subscript access for a specific integer index.
  Safe, structured alternative to string output adapters for tuple destructuring.

  Transformation:
      Input node: `func(...)`
      Output: `func(...)[index]`

  Args:
      inner_node (cst.CSTNode): The expression node (usually a Call) to slice.
      index (int): The integer index to access.

  Returns:
      cst.Subscript: The wrapped node.
  """
  # Safe conversion to string for Integer node construction
  idx_node = cst.Integer(str(index))

  return cst.Subscript(
    value=inner_node,  # type: ignore (BaseExpression)
    slice=[cst.SubscriptElement(slice=cst.Index(value=idx_node))],
  )


def rewrite_as_inline_lambda(lambda_str: str, args: list[cst.Arg]) -> cst.Call:
  """
  Wraps arguments in an Immediately Invoked Lambda Expression (IIFE).
  """
  try:
    parsed_expr = cst.parse_expression(lambda_str)
    parenthesized_lambda = parsed_expr.with_changes(lpar=[cst.LeftParen()], rpar=[cst.RightParen()])
    return cst.Call(func=parenthesized_lambda, args=args)
  except cst.ParserSyntaxError:
    raise ValueError(f"Invalid lambda syntax in semantics: {lambda_str}")


def rewrite_as_macro(template: str, args_list: list[cst.Arg], std_arg_names: list[str]) -> cst.CSTNode:
  """
  Replaces an operation call with a python expression defined in the template.

  Arguments are substituted into the template string using placeholders matching
  the standard argument names (e.g. `{x}`).

  Args:
      template (str): The macro string (e.g. "{x} * jax.nn.sigmoid({x})").
      args_list (list[cst.Arg]): The normalized argument nodes for this call.
      std_arg_names (list[str]): The names of standard arguments in order.

  Returns:
      cst.CSTNode: The constructed expression logic.

  Raises:
      ValueError: If arguments required by the template are missing.
      cst.ParserSyntaxError: If the resulting string is invalid Python.
  """
  # 1. Map args
  arg_map = {}

  # Basic positional mapping. Logic assumes `args_list` from NormalizationMixin
  # is positional aligned with `std_args` or has correct keywords.
  # However, normalization output is a list of cst.Arg objects.
  # Some might be positional (aligned), some keywords.

  # We iterate over the STANDARD names. We look for a match in args_list.
  # NormalizationMixin usually produces a list in std_args order.

  # Robust mapping strategy:
  # 1. Zip positional args with std names.
  # 2. Extract keyword args into map.

  for i, (std_name, arg) in enumerate(zip(std_arg_names, args_list)):
    # We assume the rewriter normalized them to position/keywords matching target expectations
    # But since macros define their own structure, they might refer to ANY arg.
    # If normalization mixin did its job, `args_list` elements correspond to `std_arg_names`
    # unless skipped/variadic?
    # Assuming normalization respects std order.

    # We need the source string of the value expression
    arg_val = arg.value
    # Capture the source code of the argument expression
    arg_map[std_name] = capture_node_source(arg_val)

  # 2. Format Template
  try:
    code = template.format(**arg_map)
  except KeyError as e:
    raise ValueError(f"Macro template requires argument {e} but it was missing/unresolvable.")

  # 3. Parse back to CST
  try:
    return cst.parse_expression(code)
  except cst.ParserSyntaxError:
    raise ValueError(f"Macro template output produced invalid python: {code}")


def rewrite_as_infix(
  _original_node: cst.Call,
  args: List[cst.Arg],
  op_symbol: str,
  std_args: List[str],
) -> Union[cst.BinaryOperation, cst.UnaryOperation]:
  """
  Transforms a functional call into an infix (binary) or prefix (unary) expression.
  """
  arity = len(std_args) if std_args else len(args)

  if arity == 1:
    if len(args) < 1:
      raise ValueError(f"Unary operator '{op_symbol}' expects 1 argument, got {len(args)}")

    unary_map = {
      "+": cst.Plus(),
      "-": cst.Minus(),
      "~": cst.BitInvert(),
      "not": cst.Not(),
    }
    cst_op = unary_map.get(op_symbol)
    if not cst_op:
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
