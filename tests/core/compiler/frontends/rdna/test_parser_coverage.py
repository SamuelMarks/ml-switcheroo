"""Tests for RDNA parser coverage."""

import pytest
from ml_switcheroo.core.compiler.frontends.rdna.parser import RdnaParser
from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaImmediate, RdnaMemory, RdnaDirective


def test_rdna_parser_empty_lines():
  """Test RDNA parser with empty lines."""
  parser = RdnaParser("v_mov_b32 v0, v1\n\n  \n")
  nodes = parser.parse().statements
  assert len(nodes) == 1


def test_rdna_parser_trivia():
  """Test trivia in parsing."""
  parser = RdnaParser("  ; comment\n  v_nop\n")
  mod = parser.parse()
  assert mod.statements[0].leading_trivia[0].text == "  "


def test_rdna_parser_errors():
  """Test parser error handling."""
  parser = RdnaParser("!")
  with pytest.raises(ValueError):
    parser.parse()


def test_rdna_parser_immediate_variants():
  """Test different numeric immediate variants."""
  # imm_num float
  parser = RdnaParser("v_add v0, 1.5")
  inst = parser.parse().statements[0]
  assert isinstance(inst.operands[1], RdnaImmediate)
  assert inst.operands[1].value == 1.5

  # neg_num float
  parser = RdnaParser("v_add v0, -1.5")
  inst = parser.parse().statements[0]
  assert inst.operands[1].value == -1.5

  # neg_hex
  parser = RdnaParser("v_add v0, -0xff")
  inst = parser.parse().statements[0]
  assert inst.operands[1].value == -255
  assert inst.operands[1].is_hex is True

  # pos_num
  parser = RdnaParser("v_add v0, +42")
  inst = parser.parse().statements[0]
  assert inst.operands[1].value == 42

  parser = RdnaParser("v_add v0, +42.5")
  inst = parser.parse().statements[0]
  assert inst.operands[1].value == 42.5

  # pos_hex
  parser = RdnaParser("v_add v0, +0x10")
  inst = parser.parse().statements[0]
  assert inst.operands[1].value == 16


def test_rdna_parser_directive_variants():
  """Test directives with different parameters."""
  parser = RdnaParser('.foo -1, -0x2, +3, +0x4, "string", bar:baz')
  d = parser.parse().statements[0]
  assert isinstance(d, RdnaDirective)
  assert d.params == ["-1", "-0x2", "+3", "+0x4", '"string"', "bar:baz"]


def test_rdna_parser_memory_variants():
  """Test memory operand variants."""
  parser = RdnaParser("v_add [v0 + 0x10]")
  inst = parser.parse().statements[0]
  assert isinstance(inst.operands[0], RdnaMemory)
  assert inst.operands[0].offset == 16

  parser = RdnaParser("v_add [v0 - 10]")
  inst = parser.parse().statements[0]
  assert isinstance(inst.operands[0], RdnaMemory)
  assert inst.operands[0].offset == -10


def test_rdna_parser_empty():
  """Test empty file."""
  parser = RdnaParser("")
  assert len(parser.parse().statements) == 0


def test_rdna_parser_only_trivia():
  """Test trivia with no statements."""
  parser = RdnaParser("  \n")
  mod = parser.parse()
  assert len(mod.statements) == 0
  assert len(mod.leading_trivia) == 0


def test_rdna_parser_modifier():
  """Test instruction modifier."""
  parser = RdnaParser("v_add_f32 v0, row_mask:0xf")
  inst = parser.parse().statements[0]
  assert inst.operands[1].name == "row_mask:0xf"


def test_rdna_parser_misc():
  """Test misc parsing logic."""
  # directive with single param list that is not a list? (not really possible but test coverage)
  pass


def test_rdna_parser_missing_coverage_extra():
  """Test for test_rdna_parser_missing_coverage_extra."""
  from ml_switcheroo.core.compiler.frontends.rdna.parser import _get_trivia, RdnaTransformer
  from lark import Token

  # line 113
  class DummyNode:
    pass

  assert _get_trivia(DummyNode()) == []

  # line 214
  transformer = RdnaTransformer()
  d = transformer.directive([Token("DOT", "."), Token("IDENTIFIER", "req"), "not_a_list"])
  assert d.params == ["not_a_list"]


def test_rdna_parser_branch_coverage():
  """Test for test_rdna_parser_branch_coverage."""
  from ml_switcheroo.core.compiler.frontends.rdna.parser import RdnaTransformer
  from lark import Token

  transformer = RdnaTransformer()

  # module with EOF_TRIVIA but no stmts (184->180)
  from ml_switcheroo.core.compiler.frontends.rdna.parser import RdnaToken

  tok = RdnaToken("EOF_TRIVIA", "")
  tok.leading_trivia = []
  mod = transformer.module([tok])
  assert len(mod.statements) == 0

  # directive with no params (203->215)
  d = transformer.directive([Token("DOT", "."), Token("IDENTIFIER", "req")])
  assert len(d.params) == 0
  d = transformer.directive([Token("DOT", "."), Token("IDENTIFIER", "req"), None])
  assert len(d.params) == 0
