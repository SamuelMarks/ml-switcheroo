"""Test suite for the Parser Gap module."""

import pytest
from ml_switcheroo.core.mlir.parser import MlirParser


def test_mlir_parser_type_comma():
  """Verifies the behavior of MLIR parser type comma."""
  parser = MlirParser('%0 = "foo.bar"() : (i32, i32) -> (i32, i32)')
  parser.parse()


def test_mlir_parser_type_single():
  """Verifies the behavior of MLIR parser type single."""
  parser = MlirParser('%0 = "foo.bar"() : (i32) -> i32')
  parser.parse()


def test_mlir_parser_region_eof():
  """Verifies the behavior of MLIR parser region eof."""
  parser = MlirParser('%0 = "foo.bar"() ({ ')
  with pytest.raises(SyntaxError):
    parser.parse()


def test_mlir_parser_region_empty():
  """Verifies the behavior of MLIR parser region empty."""
  parser = MlirParser('%0 = "foo.bar"() ({^bb0: }) : () -> ()')
  with pytest.raises(SyntaxError):
    parser.parse()
