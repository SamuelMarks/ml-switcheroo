import pytest
from ml_switcheroo.core.mlir.parser import MlirParser


def test_mlir_parser_type_comma():
  parser = MlirParser('%0 = "foo.bar"() : (i32, i32) -> (i32, i32)')
  ast = parser.parse()


def test_mlir_parser_type_single():
  parser = MlirParser('%0 = "foo.bar"() : (i32) -> i32')
  ast = parser.parse()


def test_mlir_parser_region_eof():
  parser = MlirParser('%0 = "foo.bar"() ({ ')
  with pytest.raises(SyntaxError):
    parser.parse()


def test_mlir_parser_region_empty():
  parser = MlirParser('%0 = "foo.bar"() ({^bb0: }) : () -> ()')
  with pytest.raises(SyntaxError):
    ast = parser.parse()
