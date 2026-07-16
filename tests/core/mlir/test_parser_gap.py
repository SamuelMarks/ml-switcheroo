"""Auto-generated doc."""

import pytest
from ml_switcheroo.core.mlir.parser import MlirParser


def test_mlir_parser_type_comma():
  """Auto-generated doc."""
  parser = MlirParser('%0 = "foo.bar"() : (i32, i32) -> (i32, i32)')
  parser.parse()


def test_mlir_parser_type_single():
  """Auto-generated doc."""
  parser = MlirParser('%0 = "foo.bar"() : (i32) -> i32')
  parser.parse()


def test_mlir_parser_region_eof():
  """Auto-generated doc."""
  parser = MlirParser('%0 = "foo.bar"() ({ ')
  with pytest.raises(SyntaxError):
    parser.parse()


def test_mlir_parser_region_empty():
  """Auto-generated doc."""
  parser = MlirParser('%0 = "foo.bar"() ({^bb0: }) : () -> ()')
  with pytest.raises(SyntaxError):
    parser.parse()
