"""Test suite for the Mlir Parser Extra2 module."""

import pytest
from ml_switcheroo.core.mlir.parser import MlirParser, Token
from ml_switcheroo.core.mlir.tokens import TokenKind


def test_cov_287():
  """Verifies the behavior of cov 287."""
  parser = MlirParser("}")
  blk = parser.parse_block(is_top_level=True)
  assert len(blk.operations) == 0


def test_cov_296_to_298():
  """Verifies the behavior of cov 296 to 298."""
  parser = MlirParser("{ \n // comment \n a = 1 }")
  parser.parse_operation()
  parser2 = MlirParser("{ \n // comment \n ^bb0: }")
  parser2.consume()
  assert parser2._is_region_start()


def test_cov_324():
  """Verifies the behavior of cov 324."""
  parser = MlirParser("{ a = 1 }")
  parser.consume()
  assert not parser._is_region_start()


def test_cov_342():
  """Verifies the behavior of cov 342."""
  parser = MlirParser("%0 \n")
  assert parser.parse_operation() is None
  parser2 = MlirParser("%0 ^bb0: ")
  assert parser2.parse_operation() is None


def test_cov_371():
  """Verifies the behavior of cov 371."""
  parser = MlirParser("%0 [ = sw.op")
  with pytest.raises(SyntaxError):
    parser.parse_operation()


def test_cov_380():
  """Verifies the behavior of cov 380."""
  parser = MlirParser("")
  parser.tokens = [
    Token(TokenKind.STRING, '"my"', 1, 1),
    Token(TokenKind.SYMBOL, ".", 1, 5),
    Token(TokenKind.IDENTIFIER, "op", 1, 6),
    Token(TokenKind.EOF, "", 1, 8),
  ]
  parser.pos = 0
  op = parser.parse_operation()
  assert op.name == '"my".op'


def test_cov_416():
  """Verifies the behavior of cov 416."""
  parser = MlirParser("sw.op { }")
  op = parser.parse_operation()
  assert len(op.attributes) == 0


def test_cov_431():
  """Verifies the behavior of cov 431."""
  parser = MlirParser("sw.op { a = [1] }")
  op = parser.parse_operation()
  assert op.attributes[0].value == "[1]"


def test_cov_534():
  """Verifies the behavior of cov 534."""
  parser = MlirParser("{ }")
  parser.parse_region()


def test_cov_543():
  """Verifies the behavior of cov 543."""
  parser = MlirParser("{ sw.op }")
  parser.parse_region()
