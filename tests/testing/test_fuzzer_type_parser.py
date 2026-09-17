"""Tests for TypeAnnotationParser."""

import libcst as cst

from ml_switcheroo.testing.fuzzer.type_parser import (
  AnyType,
  CallableType,
  DictType,
  ListType,
  NoneType,
  OptionalType,
  ParsedType,
  PrimitiveType,
  TensorType,
  TupleType,
  TypeAnnotationParser,
  UnionType,
  parse_type_annotation,
)


def test_empty_and_any() -> None:
  """Docstring."""
  assert parse_type_annotation("") == AnyType()
  assert parse_type_annotation("   ") == AnyType()
  assert parse_type_annotation("Any") == AnyType()


def test_none() -> None:
  """Docstring."""
  assert parse_type_annotation("None") == NoneType()
  assert parse_type_annotation("NoneType") == NoneType()


def test_primitives() -> None:
  """Docstring."""
  primitives: list[str] = ["int", "integer", "float", "double", "number", "bool", "boolean", "str", "string"]
  for p in primitives:
    assert parse_type_annotation(p) == PrimitiveType(name=p)


def test_tensors() -> None:
  """Docstring."""
  assert parse_type_annotation("Array") == TensorType(dims=None)
  assert parse_type_annotation("Tensor") == TensorType(dims=None)
  assert parse_type_annotation("Array[1:2]") == TensorType(dims=None)
  assert parse_type_annotation("ndarray") == TensorType(dims=None)
  assert parse_type_annotation("np.ndarray") == TensorType(dims=None)

  # Tensor with dims
  assert parse_type_annotation("Array['N']") == TensorType(dims=["N"])
  assert parse_type_annotation("Array['B', 'N']") == TensorType(dims=["B", "N"])
  assert parse_type_annotation("Tensor['N', 'M']") == TensorType(dims=["N", "M"])
  assert parse_type_annotation("np.ndarray['N']") == TensorType(dims=["N"])
  assert parse_type_annotation("Array[N, M]") == TensorType(dims=["N", "M"])


def test_callables() -> None:
  """Docstring."""
  assert parse_type_annotation("Callable") == CallableType()
  assert parse_type_annotation("func") == CallableType()
  assert parse_type_annotation("function") == CallableType()


def test_lists() -> None:
  """Docstring."""
  assert parse_type_annotation("List") == ListType(inner=AnyType())
  assert parse_type_annotation("Sequence") == ListType(inner=AnyType())
  assert parse_type_annotation("List[int]") == ListType(inner=PrimitiveType(name="int"))
  assert parse_type_annotation("List[List[float]]") == ListType(inner=ListType(inner=PrimitiveType(name="float")))


def test_dicts() -> None:
  """Docstring."""
  assert parse_type_annotation("Dict") == DictType(key_type=AnyType(), value_type=AnyType())
  assert parse_type_annotation("Mapping") == DictType(key_type=AnyType(), value_type=AnyType())
  assert parse_type_annotation("Dict[str, int]") == DictType(
    key_type=PrimitiveType(name="str"), value_type=PrimitiveType(name="int")
  )
  assert parse_type_annotation("Dict[str]") == DictType(key_type=AnyType(), value_type=AnyType())


def test_tuples() -> None:
  """Docstring."""
  assert parse_type_annotation("Tuple") == TupleType(elements=[AnyType()], variadic=True)
  assert parse_type_annotation("Tuple[int]") == TupleType(elements=[PrimitiveType(name="int")], variadic=False)
  assert parse_type_annotation("Tuple[int, float]") == TupleType(
    elements=[PrimitiveType(name="int"), PrimitiveType(name="float")], variadic=False
  )
  assert parse_type_annotation("Tuple[int, ...]") == TupleType(elements=[PrimitiveType(name="int")], variadic=True)


def test_optional() -> None:
  """Docstring."""
  assert parse_type_annotation("Optional") == OptionalType(inner=AnyType())
  assert parse_type_annotation("Optional[int]") == OptionalType(inner=PrimitiveType(name="int"))
  assert parse_type_annotation("Optional[List[str]]") == OptionalType(inner=ListType(inner=PrimitiveType(name="str")))


def test_union() -> None:
  """Docstring."""
  # Union syntax (PEP 604)
  assert parse_type_annotation("int | float") == UnionType(types=[PrimitiveType(name="int"), PrimitiveType(name="float")])
  assert parse_type_annotation("int | float | str") == UnionType(
    types=[PrimitiveType(name="int"), PrimitiveType(name="float"), PrimitiveType(name="str")]
  )
  assert parse_type_annotation("int | (float | str)") == UnionType(
    types=[PrimitiveType(name="int"), PrimitiveType(name="float"), PrimitiveType(name="str")]
  )
  # typing.Union
  assert parse_type_annotation("Union[int, float]") == UnionType(
    types=[PrimitiveType(name="int"), PrimitiveType(name="float")]
  )


def test_syntax_error_fallback() -> None:
  """Docstring."""
  # Invalid Python syntax, will fallback to PrimitiveType with raw string
  assert parse_type_annotation("a b c") == PrimitiveType(name="a b c")


def test_unknown_types() -> None:
  """Docstring."""
  assert parse_type_annotation("CustomClass") == PrimitiveType(name="CustomClass")
  assert parse_type_annotation("module.Class") == PrimitiveType(name="module.Class")


def test_generic_visit() -> None:
  """Docstring."""
  import libcst as cst

  parser: TypeAnnotationParser = TypeAnnotationParser()
  # Pass an arbitrary CST node that shouldn't normally be hit
  res: ParsedType = parser.visit(cst.Pass())
  assert isinstance(res, PrimitiveType) and getattr(res, "name") == "Unknown"


def test_binop_unknown() -> None:
  """Docstring."""
  # E.g., int + float
  res: ParsedType = parse_type_annotation("int + float")
  assert isinstance(res, PrimitiveType) and getattr(res, "name") == "Unknown"


def test_full_name() -> None:
  """Docstring."""
  import libcst as cst

  parser: TypeAnnotationParser = TypeAnnotationParser()
  assert parser._get_full_name(cst.Pass()) == ""


def test_complex_nesting() -> None:
  """Docstring."""
  type_str: str = "Optional[Dict[str, List[Tuple[int, float]]]]"
  expected: OptionalType = OptionalType(
    inner=DictType(
      key_type=PrimitiveType(name="str"),
      value_type=ListType(
        inner=TupleType(elements=[PrimitiveType(name="int"), PrimitiveType(name="float")], variadic=False)
      ),
    )
  )
  assert parse_type_annotation(type_str) == expected


def test_slice_subscript() -> None:
  """Docstring."""
  # In 'Array[1:2]', 1:2 is a cst.Slice, not cst.Index
  res: ParsedType = parse_type_annotation("Array[1:2]")
  assert isinstance(res, TensorType)
  assert getattr(res, "dims") is None


def test_parsed_type_base() -> None:
  """Docstring."""
  pt: ParsedType = ParsedType()
  assert isinstance(pt, ParsedType)


def test_cst_formatting_preservation() -> None:
  """Docstring."""
  import libcst as cst

  type_str: str = "Optional[  Dict[ str ,  int ]  ]"
  parsed: ParsedType = parse_type_annotation(type_str)

  assert getattr(parsed, "cst_node") is not None
  # Extract code directly using Module
  module: cst.Module = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=getattr(parsed, "cst_node"))])])
  # Module adds newline for SimpleStatementLine
  assert getattr(module, "code").strip() == type_str


def test_cst_fallback_formatting() -> None:
  """Docstring."""
  type_str: str = "int + float"
  parsed: ParsedType = parse_type_annotation(type_str)
  assert getattr(parsed, "cst_node") is not None
  # wait, my parser parses 'int + float' as a BinaryOperation and returns Unknown
  # the cst_node is set!
  module: cst.Module = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=getattr(parsed, "cst_node"))])])
  assert getattr(module, "code").strip() == type_str


def test_subscript_slice_non_expression() -> None:
  """Test _process_slice_elt with non-Index, non-BaseExpression element."""
  parser: TypeAnnotationParser = TypeAnnotationParser()
  dummy_subscript: cst.Subscript = cst.Subscript(
    value=cst.Name("Array"),
    slice=[cst.SubscriptElement(slice=cst.Pass())],
  )
  res: ParsedType = parser.visit_Subscript(dummy_subscript)
  assert isinstance(res, TensorType)


def test_subscript_slice_direct_expression_and_custom() -> None:
  """Test _process_slice_elt with direct BaseExpression element and custom generic subscript."""
  parser: TypeAnnotationParser = TypeAnnotationParser()
  node = cst.Subscript(
    value=cst.Name("Array"),
    slice=[cst.SubscriptElement(slice=cst.Name("N"))],
  )
  res = parser.visit_Subscript(node)
  assert res == TensorType(dims=["N"])

  # Custom generic subscript (line 321: else: res = base)
  custom_res = parse_type_annotation("CustomType[int]")
  assert isinstance(custom_res, PrimitiveType)
  assert custom_res.name == "CustomType"

  # Integer constant (lines 221-222)
  int_const = parser.visit(cst.Integer("42"))
  assert isinstance(int_const, PrimitiveType)
  assert int_const.name == "42"

  # Node without value attribute (branch 221->224)
  no_val_const = parser.visit_Constant(cst.Pass())
  assert isinstance(no_val_const, PrimitiveType)
  assert no_val_const.name == ""
