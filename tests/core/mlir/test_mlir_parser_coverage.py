"""Test module."""

import typing

import pytest

from ml_switcheroo.core.mlir.parser import MlirLexer, MlirParser


def test_mlir_parser_invalid_token() -> None:
  """Docstring."""
  with pytest.raises(ValueError, match="Unexpected"):
    MlirParser("~").parse()


def test_mlir_parser_sym_id() -> None:
  """Docstring."""
  code: str = "func.func @main() { return }"
  parser = MlirParser(code)
  module: typing.Any = parser.parse()
  assert module.body.operations[0].name == "func.func"
  assert module.body.operations[0].name_trivia[-1].text == "@main"


def test_mlir_parser_array_attr() -> None:
  """Docstring."""
  code: str = "sw.op {arr = [1, 2]}"
  parser = MlirParser(code)
  module: typing.Any = parser.parse()
  assert module.body.operations[0].attributes[0].value == ["1", "2"]


def test_mlir_parser_empty() -> None:
  """Docstring."""
  parser = MlirParser("   ")
  module: typing.Any = parser.parse()
  assert len(module.body.operations) == 0


def test_mlir_parser_op_tail_region() -> None:
  """Docstring."""
  code: str = "sw.op { ^bb0: }"
  parser = MlirParser(code)
  module: typing.Any = parser.parse()
  assert len(module.body.operations[0].regions) == 1


def test_mlir_parser_branch_coverage() -> None:
  """Docstring."""
  from ml_switcheroo.core.mlir.parser import MlirTransformer

  transformer = MlirTransformer()
  # operation with all None children (193->198)
  op: typing.Any = transformer.operation([None, None])
  assert op.name == ""

  # operation with an empty list child (229->239)
  op = transformer.operation([[]])
  assert op.name == ""


# --- Merged from test_mlir_parser_coverage_extra_loop.py ---


def test_mlir_parser_lexer_state_object():
  """Docstring."""

  class DummyLexerState:
    def __init__(self, text):
      self.text = text

  ls = DummyLexerState('%0 = "dummy"()')
  lexer = MlirLexer(None)  # Assuming lexer_conf is None is ok for this
  list(lexer.lex(ls, None))
