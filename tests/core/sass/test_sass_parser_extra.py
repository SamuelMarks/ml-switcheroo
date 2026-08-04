"""Test suite for the Sass Parser Extra module."""

import pytest

from ml_switcheroo.core.compiler.frontends.sass.parser import SassParser


def test_sass_parser_missing():
  """Verifies the behavior of SASS parser missing."""
  parser = SassParser(".text\n.global main")
  parser.parse().statements
  parser = SassParser("MOV R0, R1\n.text")
  parser.parse().statements


def test_sass_parser_comments_and_empty():
  """Tests comments and empty lines."""
  parser = SassParser("// comment\n  \n;  \n")
  nodes = parser.parse().statements
  assert len(nodes) == 1
  assert nodes[0].text == "comment"


def test_sass_parser_directive_params():
  """Tests directives with multiple params."""
  parser = SassParser(".reqntid 128, 1, 1;")
  nodes = parser.parse().statements
  assert nodes[0].name == "reqntid"
  assert nodes[0].params == ["128", "1", "1"]


def test_sass_parser_predicate_and_labels():
  """Tests predicates and labels."""
  parser = SassParser("@!P0 BRA label_target;\nlabel_target:\n@PT NOP;")
  nodes = parser.parse().statements
  assert len(nodes) == 3
  assert nodes[0].opcode == "BRA"
  assert nodes[0].predicate.name == "P0"
  assert nodes[0].predicate.negated is True
  assert nodes[1].name == "label_target"
  assert nodes[2].predicate.name == "PT"
  assert nodes[2].predicate.negated is False


def test_sass_parser_memory():
  """Tests memory operands."""
  parser = SassParser("LDG.E R0, [R1 + 0x10];\nSTG.E [R2], R3;")
  nodes = parser.parse().statements
  assert nodes[0].operands[1].base.name == "R1"
  assert nodes[0].operands[1].offset == 16
  assert nodes[1].operands[0].base.name == "R2"
  assert nodes[1].operands[0].offset is None


def test_sass_parser_constant_memory():
  """Tests constant memory operands."""
  parser = SassParser("MOV R0, c[0x0][0x120];\nMOV R1, c[0x1];")
  nodes = parser.parse().statements
  assert nodes[0].operands[1].base == "c[0x0]"
  assert nodes[0].operands[1].offset == 288
  assert nodes[1].operands[1].base == "c[0x1]"
  assert nodes[1].operands[1].offset is None


def test_sass_parser_immediates_and_registers():
  """Tests immediate and register variations."""
  parser = SassParser("FADD R0, -R1, |R2|;\nFMUL R3, |-R4|, 1.5;\nMOV R5, 0x3f800000;\nMOV R6, -0x10;\n")
  nodes = parser.parse().statements
  # FADD
  assert nodes[0].operands[1].negated is True
  assert nodes[0].operands[2].absolute is True
  # FMUL
  assert nodes[1].operands[1].negated is True
  assert nodes[1].operands[1].absolute is True
  assert nodes[1].operands[2].value == 1.5
  # MOV
  assert nodes[2].operands[1].value == 0x3F800000
  assert nodes[2].operands[1].is_hex is True
  # MOV neg hex
  assert nodes[3].operands[1].value == -16
  assert nodes[3].operands[1].is_hex is True


def test_sass_parser_missing_semicolon_and_whitespace():
  """Tests EOF handling."""
  parser = SassParser("MOV R0, R1   ")
  nodes = parser.parse().statements
  assert nodes[0].opcode == "MOV"
  assert not nodes[0].trailing_trivia


def test_sass_parser_operand_predicates():
  """Tests predicate operands."""
  parser = SassParser("ISETP.NE.AND P0, !P1, PT;")
  nodes = parser.parse().statements
  assert nodes[0].operands[0].name == "P0"
  assert nodes[0].operands[1].name == "P1"
  assert nodes[0].operands[1].negated is True
  assert nodes[0].operands[2].name == "PT"


def test_sass_parser_malformed_memory_and_infinite_loop_prevention():
  """Tests malformed memory raises ValueError with Lark parser."""
  with pytest.raises(ValueError):
    SassParser("LDG [R1 junk];\nMOV R0, c[0x0 junk];\nMOV R1, c[0x0][0x4 junk];\nMOV R2, .;\n").parse()


def test_sass_parser_empty_and_semicolon():
  """Tests edge cases for early returns."""
  assert SassParser("").parse().statements == []
  assert SassParser(";").parse().statements == []
  assert SassParser("   \n ").parse().statements == []
  # cover line 266 (peek returns None at EOF)
  assert SassParser("c").parse().statements[0].opcode == "c"
  with pytest.raises(ValueError):
    SassParser("/").parse()


def test_sass_parser_instruction_break_conditions():
  """Tests breaking instruction parsing early."""
  parser = SassParser("MOV R0, R1 // comment here\nMOV R2, R3;")
  nodes = parser.parse().statements
  assert nodes[0].opcode == "MOV"
  assert len(nodes[0].operands) == 2
  with pytest.raises(ValueError):
    SassParser("MOV R0, \n").parse()
