"""Type Annotation AST Parser.

This module provides a robust AST-based parser for Python type annotations
used in the fuzzing engine. It replaces brittle regex-based string matching
and standard `ast.parse` with `libcst` to enable safe, format-preserving
automated refactoring of type hints.
"""

import libcst as cst
from dataclasses import dataclass
from typing import List, Optional, Any


class ParsedType:
  """Base class for parsed type annotations."""

  cst_node: Optional[cst.CSTNode] = None


@dataclass
class AnyType(ParsedType):
  """Represents 'Any' or unknown types."""

  pass


@dataclass
class NoneType(ParsedType):
  """Represents 'None'."""

  pass


@dataclass
class PrimitiveType(ParsedType):
  """Represents a primitive type like int, float, str, bool.

  Attributes:
      name: The string name of the primitive type.
  """

  name: str


@dataclass
class UnionType(ParsedType):
  """Represents a Union type (e.g., A | B or Union[A, B]).

  Attributes:
      types: A list of ParsedType elements that comprise the union.
  """

  types: List[ParsedType]


@dataclass
class OptionalType(ParsedType):
  """Represents an Optional type (e.g., Optional[T]).

  Attributes:
      inner: The ParsedType that is wrapped as optional.
  """

  inner: ParsedType


@dataclass
class TupleType(ParsedType):
  """Represents a Tuple type.

  Attributes:
      elements: A list of ParsedType elements within the tuple.
      variadic: A boolean flag indicating if the tuple is variadic.
  """

  elements: List[ParsedType]
  variadic: bool


@dataclass
class ListType(ParsedType):
  """Represents a List or Sequence type.

  Attributes:
      inner: The ParsedType representing the list element type.
  """

  inner: ParsedType


@dataclass
class DictType(ParsedType):
  """Represents a Dict or Mapping type.

  Attributes:
      key_type: The ParsedType representing the dictionary keys.
      value_type: The ParsedType representing the dictionary values.
  """

  key_type: ParsedType
  value_type: ParsedType


@dataclass
class TensorType(ParsedType):
  """Represents an Array or Tensor type, optionally with symbolic dimensions.

  Attributes:
      dims: A list of strings representing the dimensions, or None.
  """

  dims: Optional[List[str]]


@dataclass
class CallableType(ParsedType):
  """Represents a Callable type."""

  pass


class TypeAnnotationParser:
  """Parses type annotation strings into ParsedType AST nodes using libcst."""

  def parse(self, type_str: str) -> ParsedType:
    """Parses a type annotation string.

    Args:
        type_str: The type annotation string (e.g., 'List[int]', 'Optional[Dict[str, Any]]').

    Returns:
        A ParsedType representing the structured type.
    """
    if not type_str or type_str.strip() == "":
      return AnyType()

    type_str = type_str.strip()

    try:
      tree = cst.parse_expression(type_str)
      return self.visit(tree)
    except cst.ParserSyntaxError:
      # Handle invalid syntax or specific string fallbacks
      return PrimitiveType(name=type_str)

  def visit(self, node: cst.CSTNode) -> ParsedType:
    """Visits a CST node and converts it to a ParsedType.

    Args:
        node: The CSTNode representing a component of the type annotation.

    Returns:
        A ParsedType representation of the visited node.
    """
    if isinstance(node, cst.Name):
      return self.visit_Name(node)
    elif isinstance(node, (cst.Integer, cst.Float, cst.SimpleString)):
      return self.visit_Constant(node)
    elif isinstance(node, cst.Attribute):
      return self.visit_Attribute(node)
    elif isinstance(node, cst.Subscript):
      return self.visit_Subscript(node)
    elif isinstance(node, cst.BinaryOperation):
      return self.visit_BinOp(node)
    elif isinstance(node, cst.Ellipsis):
      res = PrimitiveType(name="Ellipsis")
      res.cst_node = node
      return res

    res = PrimitiveType(name="Unknown")
    res.cst_node = node
    return res

  def visit_Name(self, node: cst.Name) -> ParsedType:
    """Visit a simple name node.

    Args:
        node: The CST Name node representing a simple type identifier.

    Returns:
        A ParsedType representing the parsed name identifier.
    """
    name = node.value
    res: ParsedType
    if name in ("Any",):
      res = AnyType()
    elif name in ("None", "NoneType"):
      res = NoneType()
    elif name in ("int", "integer", "float", "double", "number", "bool", "boolean", "str", "string"):
      res = PrimitiveType(name=name)
    elif name in ("Array", "Tensor", "ndarray"):
      res = TensorType(dims=None)
    elif name in ("Callable", "func", "function"):
      res = CallableType()
    elif name in ("List", "Sequence"):
      res = ListType(inner=AnyType())
    elif name in ("Dict", "Mapping"):
      res = DictType(key_type=AnyType(), value_type=AnyType())
    elif name in ("Tuple",):
      res = TupleType(elements=[AnyType()], variadic=True)
    elif name in ("Optional",):
      res = OptionalType(inner=AnyType())
    else:
      res = PrimitiveType(name=name)

    res.cst_node = node
    return res

  def visit_Constant(self, node: cst.CSTNode) -> ParsedType:
    """Visit a constant node.

    Args:
        node: The CSTNode representing a constant literal expression.

    Returns:
        A PrimitiveType capturing the constant value.
    """
    val = ""
    if isinstance(node, cst.SimpleString):
      val = node.value.strip("'\"")
    elif hasattr(node, "value"):  # pragma: no branch
      val = str(getattr(node, "value"))

    res = PrimitiveType(name=val)
    res.cst_node = node
    return res

  def visit_Attribute(self, node: cst.Attribute) -> ParsedType:
    """Visit an attribute node (e.g., np.ndarray).

    Args:
        node: The CST Attribute node representing a dotted attribute access.

    Returns:
        A ParsedType representing the attribute path.
    """
    full_name = self._get_full_name(node)
    res: ParsedType
    if full_name == "np.ndarray":
      res = TensorType(dims=None)
    else:
      res = PrimitiveType(name=full_name)
    res.cst_node = node
    return res

  def _get_full_name(self, node: cst.CSTNode) -> str:
    """Get the full dotted name string recursively from a Name or Attribute node.

    Args:
        node: The CSTNode representation of a Name or Attribute path.

    Returns:
        The full dotted name as a string.
    """
    if isinstance(node, cst.Name):
      return node.value
    elif isinstance(node, cst.Attribute):
      return f"{self._get_full_name(node.value)}.{node.attr.value}"
    return ""

  def visit_Subscript(self, node: cst.Subscript) -> ParsedType:
    """Visit a subscripted node (e.g., List[int]).

    Args:
        node: The CST Subscript node representing a generic or parameterized type.

    Returns:
        A ParsedType containing the base type and its parameterized arguments.
    """
    base = self.visit(node.value)

    args: List[ParsedType] = []
    is_variadic = False
    raw_dims: List[str] = []

    def _process_slice_elt(elt: cst.CSTNode) -> None:
      """Process a subscript slice element inside a Subscript node.

      Args:
          elt: The CSTNode representing the subscript slice element to process.
      """
      nonlocal is_variadic

      # In libcst, node.slice is a list of SubscriptElement
      # elt is the slice object inside SubscriptElement
      val: Any
      if isinstance(elt, cst.Index):
        val = elt.value
      else:
        val = elt

      if isinstance(val, cst.Ellipsis):
        is_variadic = True
      elif isinstance(val, cst.SimpleString):
        raw_dims.append(val.value.strip("'\""))
      elif isinstance(val, cst.Name) and val.value != "Ellipsis":
        raw_dims.append(val.value)

      args.append(self.visit(val))

    for elt in node.slice:
      _process_slice_elt(elt.slice)

    res: ParsedType
    if isinstance(base, OptionalType):
      res = OptionalType(inner=args[0] if args else AnyType())
    elif isinstance(base, ListType):
      res = ListType(inner=args[0] if args else AnyType())
    elif isinstance(base, TupleType):
      elements = [a for a in args if not (isinstance(a, PrimitiveType) and a.name == "Ellipsis")]
      res = TupleType(elements=elements, variadic=is_variadic)
    elif isinstance(base, DictType):
      if len(args) == 2:
        res = DictType(key_type=args[0], value_type=args[1])
      else:
        res = DictType(key_type=AnyType(), value_type=AnyType())
    elif isinstance(base, TensorType):
      res = TensorType(dims=raw_dims if raw_dims else None)
    elif isinstance(base, PrimitiveType) and base.name == "Union":
      res = UnionType(types=args)
    else:
      res = base

    res.cst_node = node
    return res

  def visit_BinOp(self, node: cst.BinaryOperation) -> ParsedType:
    """Visit a binary operation (e.g., A | B for Unions).

    Args:
        node: The CST BinaryOperation node representing the operation.

    Returns:
        A ParsedType containing UnionType if BitOr, else PrimitiveType("Unknown").
    """
    if isinstance(node.operator, cst.BitOr):
      left = self.visit(node.left)
      right = self.visit(node.right)

      types = []
      if isinstance(left, UnionType):
        types.extend(left.types)
      else:
        types.append(left)

      if isinstance(right, UnionType):
        types.extend(right.types)
      else:
        types.append(right)

      res = UnionType(types=types)
      res.cst_node = node
      return res

    unknown_res = PrimitiveType(name="Unknown")
    unknown_res.cst_node = node
    return unknown_res


def parse_type_annotation(type_str: str) -> ParsedType:
  """Helper function to parse a type annotation string.

  Args:
      type_str: The type annotation string.

  Returns:
      The ParsedType representation.
  """
  parser = TypeAnnotationParser()
  return parser.parse(type_str)
