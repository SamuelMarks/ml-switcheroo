"""Test suite for the Mlir Parser Regions module."""

import pytest
from ml_switcheroo.core.mlir.parser_regions import MlirParserRegionsMixin
from ml_switcheroo.core.mlir.parser_base import MlirParserBase
from ml_switcheroo.core.mlir.tokens import TokenKind


class DummyParser(MlirParserRegionsMixin, MlirParserBase):
  """Dummy Parser class for testing purposes."""

  def parse_type(self):
    """Mock implementation of parse type."""
    tk = self.peek()
    if tk.kind != TokenKind.EOF and tk.text != "," and (tk.text != ")"):
      self.consume()
    return "Type"

  def parse_operation(self):
    """Mock implementation of parse operation."""
    tk = self.peek()
    if tk.text == "op":
      self.consume()
      return "OpNode"
    if tk.kind != TokenKind.EOF and tk.kind != TokenKind.BLOCK_LABEL and (tk.text != "}"):
      self.consume()
    return None


def test_expect():
  """Verifies the behavior of expect."""
  parser = DummyParser("foo")
  with pytest.raises(SyntaxError):
    parser._expect("bar")
  tk = parser._expect("foo")
  assert tk.text == "foo"


def test_absorb_trivia():
  """Verifies the behavior of absorb trivia."""
  parser = DummyParser(" \n // hello \n op")
  parser._absorb_trivia()
  assert parser.peek().text == "op"
  trivia = parser._flush_trivia()
  assert len(trivia) == 6


def test_absorb_trivia_eof():
  """Verifies the behavior of absorb trivia eof."""
  parser = DummyParser(" ")
  parser._absorb_trivia()
  assert parser.peek().kind == TokenKind.EOF


def test_parse_module():
  """Parses module."""
  parser = DummyParser("op op")
  mod = parser.parse()
  assert mod is not None
  assert len(mod.body.operations) == 2


def test_parse_block_label():
  """Parses block label."""
  parser = DummyParser("^bb0(%val : Type): op")
  block = parser.parse_block()
  assert block.label == "^bb0"
  assert len(block.arguments) == 1
  assert block.arguments[0][0].name == "%val"
  assert block.arguments[0][1] == "Type"
  assert len(block.operations) == 1


def test_parse_block_label_no_colon():
  """Parses block label no colon."""
  parser = DummyParser("^bb0(%val : Type) op")
  block = parser.parse_block()
  assert block.label == "^bb0"


def test_parse_block_multi_args():
  """Parses block multi arguments."""
  parser = DummyParser("^bb0(%v1 : Type, %v2 : Type): op")
  block = parser.parse_block()
  assert len(block.arguments) == 2


def test_parse_block_empty():
  """Parses block empty."""
  parser = DummyParser("}")
  block = parser.parse_block()
  assert len(block.operations) == 0


def test_parse_region():
  """Parses region."""
  parser = DummyParser("{ ^bb0: op }")
  region = parser.parse_region()
  assert len(region.blocks) == 1
  assert region.blocks[0].label == "^bb0"


def test_parse_region_implicit_block():
  """Parses region implicit block."""
  parser = DummyParser("{ op ^bb1: op }")
  region = parser.parse_region()
  assert len(region.blocks) == 2
  assert region.blocks[0].label == ""
  assert region.blocks[1].label == "^bb1"


def test_parse_region_no_braces():
  """Parses region no braces."""
  parser = DummyParser("^bb0: op")
  region = parser.parse_region()
  assert len(region.blocks) == 1


def test_parse_region_eof():
  """Parses region eof."""
  parser = DummyParser("{")
  region = parser.parse_region()
  assert len(region.blocks) == 0


def test_parse_region_early_exit():
  """Parses region early exit."""
  parser = DummyParser("{ op something_else }")
  region = parser.parse_region()
  assert len(region.blocks) == 1
