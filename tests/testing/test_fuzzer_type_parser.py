"""Tests for TypeAnnotationParser."""

from unittest.mock import MagicMock

import libcst as _cst

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

cst = _cst


def test_empty_and_any() -> None:
  """Verify `parse_type_annotation` handles empty and any as expected."""
  assert parse_type_annotation("") == AnyType()
  assert parse_type_annotation("   ") == AnyType()
  assert parse_type_annotation("Any") == AnyType()


def test_none() -> None:
  """Verify `parse_type_annotation` handles none as expected."""
  assert parse_type_annotation("None") == NoneType()
  assert parse_type_annotation("NoneType") == NoneType()


def test_primitives() -> None:
  """Verify `parse_type_annotation` handles primitives as expected."""
  primitives = ["int", "integer", "float", "double", "number", "bool", "boolean", "str", "string"]
  for p in primitives:
    assert parse_type_annotation(p) == PrimitiveType(name=p)


def test_tensors() -> None:
  """Verify `parse_type_annotation` handles tensors as expected."""
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
  """Verify `parse_type_annotation` handles callables as expected."""
  assert parse_type_annotation("Callable") == CallableType()
  assert parse_type_annotation("func") == CallableType()
  assert parse_type_annotation("function") == CallableType()


def test_lists() -> None:
  """Verify `parse_type_annotation` handles lists as expected."""
  assert parse_type_annotation("List") == ListType(inner=AnyType())
  assert parse_type_annotation("Sequence") == ListType(inner=AnyType())
  assert parse_type_annotation("List[int]") == ListType(inner=PrimitiveType(name="int"))
  assert parse_type_annotation("List[List[float]]") == ListType(inner=ListType(inner=PrimitiveType(name="float")))


def test_dicts() -> None:
  """Verify `parse_type_annotation` handles dicts as expected."""
  assert parse_type_annotation("Dict") == DictType(key_type=AnyType(), value_type=AnyType())
  assert parse_type_annotation("Mapping") == DictType(key_type=AnyType(), value_type=AnyType())
  assert parse_type_annotation("Dict[str, int]") == DictType(
    key_type=PrimitiveType(name="str"), value_type=PrimitiveType(name="int")
  )
  assert parse_type_annotation("Dict[str]") == DictType(key_type=AnyType(), value_type=AnyType())


def test_tuples() -> None:
  """Verify `parse_type_annotation` handles tuples as expected."""
  assert parse_type_annotation("Tuple") == TupleType(elements=[AnyType()], variadic=True)
  assert parse_type_annotation("Tuple[int]") == TupleType(elements=[PrimitiveType(name="int")], variadic=False)
  assert parse_type_annotation("Tuple[int, float]") == TupleType(
    elements=[PrimitiveType(name="int"), PrimitiveType(name="float")], variadic=False
  )
  assert parse_type_annotation("Tuple[int, ...]") == TupleType(elements=[PrimitiveType(name="int")], variadic=True)


def test_optional() -> None:
  """Verify `parse_type_annotation` handles optional as expected."""
  assert parse_type_annotation("Optional") == OptionalType(inner=AnyType())
  assert parse_type_annotation("Optional[int]") == OptionalType(inner=PrimitiveType(name="int"))
  assert parse_type_annotation("Optional[List[str]]") == OptionalType(inner=ListType(inner=PrimitiveType(name="str")))


def test_union() -> None:
  """Verify `parse_type_annotation` handles union as expected."""
  # Union syntax (PEP 604)
  assert parse_type_annotation("int | float") == UnionType(types=[PrimitiveType(name="int"), PrimitiveType(name="float")])
  assert parse_type_annotation("int | float | str") == UnionType(
    types=[PrimitiveType(name="int"), PrimitiveType(name="float"), PrimitiveType(name="str")]
  )
  assert parse_type_annotation("int | (float | str)") == UnionType(
    types=[PrimitiveType(name="int"), PrimitiveType(name="float"), PrimitiveType(name="str")]
  )
  # Union[...] subscript syntax
  assert parse_type_annotation("Union[int, float]") == UnionType(
    types=[PrimitiveType(name="int"), PrimitiveType(name="float")]
  )


def test_syntax_error_fallback() -> None:
  """Verify `parse_type_annotation` handles syntax error fallback as expected."""
  # Invalid Python syntax, will fallback to PrimitiveType with raw string
  assert parse_type_annotation("a b c") == PrimitiveType(name="a b c")


def test_unknown_types() -> None:
  """Verify `parse_type_annotation` handles unknown types as expected."""
  assert parse_type_annotation("CustomClass") == PrimitiveType(name="CustomClass")
  assert parse_type_annotation("module.Class") == PrimitiveType(name="module.Class")


def test_generic_visit() -> None:
  """Verify `TypeAnnotationParser` handles generic visit as expected."""
  import libcst as cst

  parser = TypeAnnotationParser()
  # Pass an arbitrary CST node that shouldn't normally be hit
  res = parser.visit(cst.Pass())
  assert isinstance(res, PrimitiveType)
  assert res.name == "Unknown"


def test_binop_unknown() -> None:
  """Verify `parse_type_annotation` handles binop unknown as expected."""
  # E.g., int + float
  res = parse_type_annotation("int + float")
  assert isinstance(res, PrimitiveType)
  assert res.name == "Unknown"


def test_full_name() -> None:
  """Verify `TypeAnnotationParser` handles full name as expected."""
  import libcst as cst

  parser = TypeAnnotationParser()
  assert parser._get_full_name(cst.Pass()) == ""


def test_complex_nesting() -> None:
  """Verify `OptionalType` handles complex nesting as expected."""
  type_str = "Optional[Dict[str, List[Tuple[int, float]]]]"
  expected = OptionalType(
    inner=DictType(
      key_type=PrimitiveType(name="str"),
      value_type=ListType(
        inner=TupleType(elements=[PrimitiveType(name="int"), PrimitiveType(name="float")], variadic=False)
      ),
    )
  )
  assert parse_type_annotation(type_str) == expected


def test_slice_subscript() -> None:
  """Verify `parse_type_annotation` handles slice subscript as expected."""
  # In 'Array[1:2]', 1:2 is a cst.Slice, not cst.Index
  res = parse_type_annotation("Array[1:2]")
  assert isinstance(res, TensorType)
  assert res.dims is None


def test_parsed_type_base() -> None:
  """Verify `ParsedType` handles parsed type base as expected."""
  pt = ParsedType()
  assert isinstance(pt, ParsedType)
  assert pt.cst_node is None
  name_node = cst.Name("int")
  pt_with_node = ParsedType(cst_node=name_node)
  assert pt_with_node.cst_node is name_node
  assert pt == pt_with_node


def test_cst_formatting_preservation() -> None:
  """Verify `parse_type_annotation` handles CST formatting preservation as expected."""
  import libcst as cst

  type_str = "Optional[  Dict[ str ,  int ]  ]"
  parsed = parse_type_annotation(type_str)

  assert isinstance(parsed.cst_node, cst.BaseExpression)
  # Extract code directly using Module
  module = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=parsed.cst_node)])])
  # Module adds newline for SimpleStatementLine
  assert module.code.strip() == type_str


def test_cst_fallback_formatting() -> None:
  """Verify `parse_type_annotation` handles CST fallback formatting as expected."""
  type_str = "int + float"
  parsed = parse_type_annotation(type_str)
  assert isinstance(parsed.cst_node, cst.BaseExpression)
  # wait, my parser parses 'int + float' as a BinaryOperation and returns Unknown
  # the cst_node is set!
  module = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=parsed.cst_node)])])
  assert module.code.strip() == type_str


def test_subscript_slice_non_expression() -> None:
  """Test _process_slice_elt with non-Index, non-BaseExpression element."""
  parser = TypeAnnotationParser()
  dummy_subscript = cst.Subscript(
    value=cst.Name("Array"),
    slice=[cst.SubscriptElement(slice=cst.Slice(lower=None, upper=None))],
  )
  res = parser.visit_Subscript(dummy_subscript)
  assert isinstance(res, TensorType)
  assert res.dims is None
  assert res == TensorType(dims=None)
  assert res.cst_node is dummy_subscript


def test_subscript_slice_direct_expression_and_custom() -> None:
  """Test _process_slice_elt with direct BaseExpression element and custom generic subscript."""
  parser = TypeAnnotationParser()
  mock_elem = MagicMock(spec=cst.SubscriptElement)
  mock_elem.slice = cst.Name("N")
  node = cst.Subscript(
    value=cst.Name("Array"),
    slice=[mock_elem],
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
