"""Test suite for the Mlir Parser Base module."""

import pytest
from ml_switcheroo.core.mlir.parser_base import MlirParserBase
from ml_switcheroo.core.mlir.tokens import TokenKind


def test_parser_base_init():
  """Verifies the behavior of parser base initialization."""
  parser = MlirParserBase("hello world")
  assert len(parser.tokens) > 0


def test_parser_base_peek():
  """Verifies the behavior of parser base peek."""
  parser = MlirParserBase("foo bar")
  assert parser.peek().text == "foo"
  assert parser.peek(2).text == "bar"


def test_parser_base_peek_eof():
  """Verifies the behavior of parser base peek eof."""
  parser = MlirParserBase("")
  assert parser.peek().kind == TokenKind.EOF
  assert parser.peek(10).kind == TokenKind.EOF


def test_parser_base_consume():
  """Verifies the behavior of parser base consume."""
  parser = MlirParserBase("foo bar")
  tk = parser.consume()
  assert tk.text == "foo"
  tk2 = parser.consume()
  assert tk2.text == " "


def test_parser_base_consume_eof():
  """Verifies the behavior of parser base consume eof."""
  parser = MlirParserBase("")
  tk = parser.consume()
  assert tk.kind == TokenKind.EOF
  tk2 = parser.consume()
  assert tk2.kind == TokenKind.EOF


def test_parser_base_match():
  """Verifies the behavior of parser base match."""
  parser = MlirParserBase("foo")
  assert parser.match(TokenKind.IDENTIFIER) is True
  assert parser.match("foo") is True
  assert parser.match("bar") is False


def test_parser_base_expect():
  """Verifies the behavior of parser base expect."""
  parser = MlirParserBase("foo")
  tk = parser.expect(TokenKind.IDENTIFIER)
  assert tk.text == "foo"


def test_parser_base_expect_fail():
  """Verifies the behavior of parser base expect fail."""
  parser = MlirParserBase("foo")
  with pytest.raises(SyntaxError, match="Expected bar"):
    parser.expect("bar")


def test_parser_base_absorb_trivia():
  """Verifies the behavior of parser base absorb trivia."""
  parser = MlirParserBase("  \n // comment \n foo")
  parser._absorb_trivia()
  assert parser.peek().text == "foo"
  trivia = parser._flush_trivia()
  assert len(trivia) == 1
  assert trivia[0].content == "// comment "


def test_parser_base_flush_trivia():
  """Verifies the behavior of parser base flush trivia."""
  parser = MlirParserBase("")
  assert parser._flush_trivia() == []
