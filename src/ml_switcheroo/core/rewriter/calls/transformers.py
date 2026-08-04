"""AST Transformation Helpers.

Provides logic to reconstruct CST nodes for Infix operators, Inline Lambdas,
and Structured Index Selection.
"""

import libcst as cst
from typing import List, Union


def apply_index_select(inner_node: cst.CSTNode, index: int) -> cst.Subscript:
  """Wraps an expression node with a subscript access for a specific integer index.


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
    value=inner_node,  # type: ignore
    slice=[cst.SubscriptElement(slice=cst.Index(value=idx_node))],
  )


def rewrite_as_inline_lambda(lambda_str: str, args: list[cst.Arg]) -> cst.Call:
  """Wraps arguments in an Immediately Invoked Lambda Expression (IIFE).

  Args:
      lambda_str (str): The string representation of the lambda function.
      args (list[cst.Arg]): The list of arguments to pass to the lambda.

  Returns:
      cst.Call: A CST call node representing the invoked lambda.

  Raises:
      ValueError: If the lambda string has invalid syntax.

  """
  try:
    parsed_expr = cst.parse_expression(lambda_str)
    parenthesized_lambda = parsed_expr.with_changes(lpar=[cst.LeftParen()], rpar=[cst.RightParen()])
    return cst.Call(func=parenthesized_lambda, args=args)
  except cst.ParserSyntaxError:
    raise ValueError(f"Invalid lambda syntax in semantics: {lambda_str}")


class MacroSubstitutionTransformer(cst.CSTTransformer):
  """Substitutes variables inside a parsed macro expression with CST nodes."""

  def __init__(self, arg_map: dict[str, cst.BaseExpression]):
    """Initialize the transformer with an argument map.

    Args:
        arg_map (dict[str, cst.BaseExpression]): A mapping from standardized argument names
            to their corresponding parsed CST expression nodes.
    """
    self.arg_map = arg_map

  def leave_Name(self, original_node: cst.Name, updated_node: cst.Name) -> cst.BaseExpression:
    """Replaces Name nodes with mapped values if they match the safe macro prefix.

    Args:
        original_node (cst.Name): The original un-modified name node.
        updated_node (cst.Name): The updated name node.

    Returns:
        cst.BaseExpression: The corresponding CST node from the argument map,
        or the original node if it was not a macro variable.
    """
    if original_node.value.startswith("_MACRO_VAR_") and original_node.value.endswith("_"):
      field_name = original_node.value[11:-1]
      if field_name in self.arg_map:
        return self.arg_map[field_name]
    return updated_node


def rewrite_as_macro(template: str, args_list: list[cst.Arg], std_arg_names: list[str]) -> cst.BaseExpression:
  """Replaces an operation call with a Python expression defined in the template.

  Arguments are substituted into the template string structurally by first parsing
  a sanitized version of the template into a Concrete Syntax Tree, and then
  replacing placeholder identifiers with the actual argument CST nodes. This
  prevents syntax injection vulnerabilities and parser crashes during expansion.

  Args:
      template (str): The macro string (e.g. "{x} * jax.nn.sigmoid({x})").
      args_list (list[cst.Arg]): The normalized argument nodes for this call.
      std_arg_names (list[str]): The names of standard arguments in order.

  Returns:
      cst.BaseExpression: The constructed expression logic.

  Raises:
      ValueError: If arguments required by the template are missing, or if the
          resulting template produces invalid Python syntax.
  """
  arg_map: dict[str, cst.BaseExpression] = {}

  for i, (std_name, arg) in enumerate(zip(std_arg_names, args_list)):
    arg_map[std_name] = arg.value

  import string

  parsed_format = list(string.Formatter().parse(template))
  clean_template = ""
  for literal_text, field_name, format_spec, conversion in parsed_format:
    clean_template += literal_text
    if field_name is not None:
      if field_name not in arg_map:
        raise ValueError(f"Macro template requires argument '{field_name}' but it was missing/unresolvable.")

      # Inject a unique safe identifier that can be reliably found in the CST
      safe_id = f"_MACRO_VAR_{field_name}_"
      clean_template += safe_id

  try:
    base_expr = cst.parse_expression(clean_template)
    from typing import cast

    return cast(cst.BaseExpression, base_expr.visit(MacroSubstitutionTransformer(arg_map)))
  except cst.ParserSyntaxError:
    raise ValueError(f"Macro template output produced invalid python: {clean_template}")


def rewrite_as_infix(
  _original_node: cst.Call,
  args: List[cst.Arg],
  op_symbol: str,
  std_args: List[str],
) -> Union[cst.BinaryOperation, cst.UnaryOperation]:
  """Transforms a functional call into an infix (binary) or prefix (unary) expression.

  Args:
      _original_node (cst.Call): The original call node.
      args (List[cst.Arg]): The arguments for the operator.
      op_symbol (str): The operator symbol (e.g., "+", "*").
      std_args (List[str]): Standard argument names for arity checking.

  Returns:
      Union[cst.BinaryOperation, cst.UnaryOperation]: The new AST node for the operation.

  Raises:
      ValueError: If unsupported operators or wrong number of arguments are passed.

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
    cst_op = op_map.get(op_symbol)  # type: ignore
    if not cst_op:
      raise ValueError(f"Unsupported binary operator: {op_symbol}")

    return cst.BinaryOperation(left=args[0].value, operator=cst_op, right=args[1].value)  # type: ignore

  else:
    raise ValueError(f"Infix operator requires 1 or 2 args, got {len(args)}")
