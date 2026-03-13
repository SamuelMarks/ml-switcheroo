"""Module docstring."""

import pytest
from ml_switcheroo.core.mlir.parser import MlirParser, Tokenizer, Token
from ml_switcheroo.core.mlir.nodes import BlockNode
from ml_switcheroo.core.mlir.tokens import TokenKind, Symbol
import re


def test_cov_287():
  """Function docstring."""
  # tk.text == Symbol.RBRACE and not is_top_level breaks at 280.
  # To hit 287, tk.text == RBRACE and is_top_level=True!
  parser = MlirParser("}")
  blk = parser.parse_block(is_top_level=True)
  assert len(blk.operations) == 0


def test_cov_296_to_298():
  """Function docstring."""
  # offset loop in _is_region_start
  # need { followed by whitespace/comments, then something else.
  parser = MlirParser("{ \n // comment \n a = 1 }")
  # it needs to be triggered from inside parse_operation!
  op = parser.parse_operation()
  # "{" a = 1 "}" is parsed as region? No, "a" has no = after identifier scan.
  # Actually just call _is_region_start directly:
  parser2 = MlirParser("{ \n // comment \n ^bb0: }")
  parser2.consume()  # consume {
  assert parser2._is_region_start() == True


def test_cov_324():
  """Function docstring."""
  # in _is_region_start, identifier followed by = -> dictionary
  parser = MlirParser("{ a = 1 }")
  parser.consume()
  assert parser._is_region_start() == False


def test_cov_342():
  """Function docstring."""
  # EOF, NEWLINE, BLOCK_LABEL in lookahead
  parser = MlirParser("%0 \n")
  assert parser.parse_operation() is None

  parser2 = MlirParser("%0 ^bb0: ")
  assert parser2.parse_operation() is None


def test_cov_371():
  """Function docstring."""
  parser = MlirParser("%0 [ = sw.op")
  with pytest.raises(SyntaxError):
    parser.parse_operation()


def test_cov_380():
  """Function docstring."""
  # op_name += "." + consume().text
  # To avoid the tokenizer error, let's inject tokens.
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
  """Function docstring."""
  # if self.match(Symbol.RBRACE): break
  parser = MlirParser("sw.op { }")
  op = parser.parse_operation()
  assert len(op.attributes) == 0


def test_cov_431():
  """Function docstring."""
  # if tk.text == Symbol.LBRACKET:
  parser = MlirParser("sw.op { a = [1] }")
  op = parser.parse_operation()
  assert op.attributes[0].value == "[1]"


def test_cov_534():
  """Function docstring."""
  # if match(RBRACE): break  in parse_region empty block defensive loop
  parser = MlirParser("{ }")
  parser.parse_region()


def test_cov_543():
  """Function docstring."""
  # if match(RBRACE): self.consume() at end of parse_region
  # This is hit when parse_region finishes and consumes }
  parser = MlirParser("{ sw.op }")
  parser.parse_region()
