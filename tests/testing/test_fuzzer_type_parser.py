"""Tests for TypeAnnotationParser."""

import libcst as cst
from ml_switcheroo.testing.fuzzer.type_parser import (
  parse_type_annotation,
  ParsedType,
  AnyType,
  NoneType,
  PrimitiveType,
  UnionType,
  OptionalType,
  TupleType,
  ListType,
  DictType,
  TensorType,
  CallableType,
  TypeAnnotationParser,
)


def test_empty_and_any():
  """Test parsing empty and Any."""
  assert parse_type_annotation("") == AnyType()
  assert parse_type_annotation("   ") == AnyType()
  assert parse_type_annotation("Any") == AnyType()


def test_none():
  """Test parsing None."""
  assert parse_type_annotation("None") == NoneType()
  assert parse_type_annotation("NoneType") == NoneType()


def test_primitives():
  """Test parsing primitives."""
  primitives = ["int", "integer", "float", "double", "number", "bool", "boolean", "str", "string"]
  for p in primitives:
    assert parse_type_annotation(p) == PrimitiveType(name=p)


def test_tensors():
  """Test parsing tensors."""
  assert parse_type_annotation("Array") == TensorType(dims=None)
  assert parse_type_annotation("Tensor") == TensorType(dims=None)
  assert parse_type_annotation("ndarray") == TensorType(dims=None)
  assert parse_type_annotation("np.ndarray") == TensorType(dims=None)

  # Tensor with dims
  assert parse_type_annotation("Array['N']") == TensorType(dims=["N"])
  assert parse_type_annotation("Array['B', 'N']") == TensorType(dims=["B", "N"])
  assert parse_type_annotation("Tensor['N', 'M']") == TensorType(dims=["N", "M"])
  assert parse_type_annotation("np.ndarray['N']") == TensorType(dims=["N"])
  assert parse_type_annotation("Array[N, M]") == TensorType(dims=["N", "M"])


def test_callables():
  """Test parsing callables."""
  assert parse_type_annotation("Callable") == CallableType()
  assert parse_type_annotation("func") == CallableType()
  assert parse_type_annotation("function") == CallableType()


def test_lists():
  """Test parsing lists."""
  assert parse_type_annotation("List") == ListType(inner=AnyType())
  assert parse_type_annotation("Sequence") == ListType(inner=AnyType())
  assert parse_type_annotation("List[int]") == ListType(inner=PrimitiveType(name="int"))
  assert parse_type_annotation("List[List[float]]") == ListType(inner=ListType(inner=PrimitiveType(name="float")))


def test_dicts():
  """Test parsing dicts."""
  assert parse_type_annotation("Dict") == DictType(key_type=AnyType(), value_type=AnyType())
  assert parse_type_annotation("Mapping") == DictType(key_type=AnyType(), value_type=AnyType())
  assert parse_type_annotation("Dict[str, int]") == DictType(
    key_type=PrimitiveType(name="str"), value_type=PrimitiveType(name="int")
  )
  assert parse_type_annotation("Dict[str]") == DictType(key_type=AnyType(), value_type=AnyType())


def test_tuples():
  """Test parsing tuples."""
  assert parse_type_annotation("Tuple") == TupleType(elements=[AnyType()], variadic=True)
  assert parse_type_annotation("Tuple[int]") == TupleType(elements=[PrimitiveType(name="int")], variadic=False)
  assert parse_type_annotation("Tuple[int, float]") == TupleType(
    elements=[PrimitiveType(name="int"), PrimitiveType(name="float")], variadic=False
  )
  assert parse_type_annotation("Tuple[int, ...]") == TupleType(elements=[PrimitiveType(name="int")], variadic=True)


def test_optional():
  """Test parsing optional."""
  assert parse_type_annotation("Optional") == OptionalType(inner=AnyType())
  assert parse_type_annotation("Optional[int]") == OptionalType(inner=PrimitiveType(name="int"))
  assert parse_type_annotation("Optional[List[str]]") == OptionalType(inner=ListType(inner=PrimitiveType(name="str")))


def test_union():
  """Test parsing unions."""
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


def test_syntax_error_fallback():
  """Test fallback on syntax error."""
  # Invalid Python syntax, will fallback to PrimitiveType with raw string
  assert parse_type_annotation("a b c") == PrimitiveType(name="a b c")


def test_unknown_types():
  """Test unknown types."""
  assert parse_type_annotation("CustomClass") == PrimitiveType(name="CustomClass")
  assert parse_type_annotation("module.Class") == PrimitiveType(name="module.Class")


def test_generic_visit():
  """Test generic visit fallback."""
  import libcst as cst

  parser = TypeAnnotationParser()
  # Pass an arbitrary CST node that shouldn't normally be hit
  res = parser.visit(cst.Pass())
  assert isinstance(res, PrimitiveType) and res.name == "Unknown"


def test_binop_unknown():
  """Test unknown binop fallback."""
  # E.g., int + float
  res = parse_type_annotation("int + float")
  assert isinstance(res, PrimitiveType) and res.name == "Unknown"


def test_full_name():
  """Test _get_full_name edge cases."""
  import libcst as cst

  parser = TypeAnnotationParser()
  assert parser._get_full_name(cst.Pass()) == ""


def test_complex_nesting():
  """Test a deeply nested structure to ensure robustness."""
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


def test_slice_subscript():
  """Test subscript with a slice to cover the non-Index path."""
  # In 'Array[1:2]', 1:2 is a cst.Slice, not cst.Index
  res = parse_type_annotation("Array[1:2]")
  assert isinstance(res, TensorType)
  assert res.dims is None


def test_parsed_type_base():
  """Test the base class."""
  pt = ParsedType()
  assert isinstance(pt, ParsedType)


def test_cst_formatting_preservation():
  """Test that libcst nodes preserve exact original strings."""
  import libcst as cst

  type_str = "Optional[  Dict[ str ,  int ]  ]"
  parsed = parse_type_annotation(type_str)

  assert parsed.cst_node is not None
  # Extract code directly using Module
  module = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=parsed.cst_node)])])
  # Module adds newline for SimpleStatementLine
  assert module.code.strip() == type_str


def test_cst_fallback_formatting():
  """Test formatting on fallback strings."""
  type_str = "int + float"
  parsed = parse_type_annotation(type_str)
  assert parsed.cst_node is not None
  # wait, my parser parses 'int + float' as a BinaryOperation and returns Unknown
  # the cst_node is set!
  module = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=parsed.cst_node)])])
  assert module.code.strip() == type_str
