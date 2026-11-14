"""Test suite for the Sass Parser Extra module."""

import pytest

from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import (
  NvidiaSassComment,
  NvidiaSassDirective,
  NvidiaSassImmediate,
  NvidiaSassInstruction,
  NvidiaSassLabel,
  NvidiaSassMemory,
  NvidiaSassPredicate,
  NvidiaSassRegister,
)
from ml_switcheroo.core.compiler.frontends.nvidia_sass.parser import NvidiaSassParser


def test_nvidia_sass_parser_missing() -> None:
  """Verify `NvidiaSassParser` returns `2` for nvidia SASS parser missing."""
  parser = NvidiaSassParser(".text\n.global main")
  assert len(parser.parse().statements) == 2
  parser = NvidiaSassParser("MOV R0, R1\n.text")
  assert len(parser.parse().statements) == 2


def test_nvidia_sass_parser_comments_and_empty() -> None:
  """Verify `NvidiaSassParser` handles nvidia SASS parser comments and empty as expected."""
  parser = NvidiaSassParser("// comment\n  \n;  \n")
  nodes = parser.parse().statements
  assert len(nodes) == 1
  stmt = nodes[0]
  assert isinstance(stmt, NvidiaSassComment)
  assert stmt.text == "comment"


def test_nvidia_sass_parser_directive_params() -> None:
  """Verify `NvidiaSassParser` handles nvidia SASS parser directive params as expected."""
  parser = NvidiaSassParser(".reqntid 128, 1, 1;")
  nodes = parser.parse().statements
  stmt = nodes[0]
  assert isinstance(stmt, NvidiaSassDirective)
  assert stmt.name == "reqntid"
  assert stmt.params == ["128", "1", "1"]


def test_nvidia_sass_parser_predicate_and_labels() -> None:
  """Verify `NvidiaSassParser` handles nvidia SASS parser predicate and labels as expected."""
  parser = NvidiaSassParser("@!P0 BRA label_target;\nlabel_target:\n@PT NOP;")
  nodes = parser.parse().statements
  assert len(nodes) == 3
  stmt0 = nodes[0]
  assert isinstance(stmt0, NvidiaSassInstruction)
  assert stmt0.opcode == "BRA"
  assert stmt0.predicate is not None
  assert stmt0.predicate.name == "P0"
  assert stmt0.predicate.negated is True
  stmt1 = nodes[1]
  assert isinstance(stmt1, NvidiaSassLabel)
  assert stmt1.name == "label_target"
  stmt2 = nodes[2]
  assert isinstance(stmt2, NvidiaSassInstruction)
  assert stmt2.predicate is not None
  assert stmt2.predicate.name == "PT"
  assert stmt2.predicate.negated is False


def test_nvidia_sass_parser_memory() -> None:
  """Verify `NvidiaSassParser` returns `16` for nvidia SASS parser memory."""
  parser = NvidiaSassParser("LDG.E R0, [R1 + 0x10];\nSTG.E [R2], R3;")
  nodes = parser.parse().statements
  stmt0 = nodes[0]
  assert isinstance(stmt0, NvidiaSassInstruction)
  mem0 = stmt0.operands[1]
  assert isinstance(mem0, NvidiaSassMemory)
  assert isinstance(mem0.base, NvidiaSassRegister)
  assert mem0.base.name == "R1"
  assert mem0.offset == 16
  stmt1 = nodes[1]
  assert isinstance(stmt1, NvidiaSassInstruction)
  mem1 = stmt1.operands[0]
  assert isinstance(mem1, NvidiaSassMemory)
  assert isinstance(mem1.base, NvidiaSassRegister)
  assert mem1.base.name == "R2"
  assert mem1.offset is None


def test_nvidia_sass_parser_constant_memory() -> None:
  """Verify `NvidiaSassParser` handles nvidia SASS parser constant memory as expected."""
  parser = NvidiaSassParser("MOV R0, c[0x0][0x120];\nMOV R1, c[0x1];")
  nodes = parser.parse().statements
  stmt0 = nodes[0]
  assert isinstance(stmt0, NvidiaSassInstruction)
  mem0 = stmt0.operands[1]
  assert isinstance(mem0, NvidiaSassMemory)
  assert mem0.base == "c[0x0]"
  assert mem0.offset == 288
  stmt1 = nodes[1]
  assert isinstance(stmt1, NvidiaSassInstruction)
  mem1 = stmt1.operands[1]
  assert isinstance(mem1, NvidiaSassMemory)
  assert mem1.base == "c[0x1]"
  assert mem1.offset is None


def test_nvidia_sass_parser_immediates_and_registers() -> None:
  """Verify `NvidiaSassParser` handles nvidia SASS parser immediates and registers as expected."""
  parser = NvidiaSassParser("FADD R0, -R1, |R2|;\nFMUL R3, |-R4|, 1.5;\nMOV R5, 0x3f800000;\nMOV R6, -0x10;\n")
  nodes = parser.parse().statements
  stmt0 = nodes[0]
  assert isinstance(stmt0, NvidiaSassInstruction)
  s0_op1 = stmt0.operands[1]
  s0_op2 = stmt0.operands[2]
  assert isinstance(s0_op1, NvidiaSassRegister)
  assert isinstance(s0_op2, NvidiaSassRegister)
  assert s0_op1.negated is True
  assert s0_op2.absolute is True
  stmt1 = nodes[1]
  assert isinstance(stmt1, NvidiaSassInstruction)
  s1_op1 = stmt1.operands[1]
  s1_op2 = stmt1.operands[2]
  assert isinstance(s1_op1, NvidiaSassRegister)
  assert isinstance(s1_op2, NvidiaSassImmediate)
  assert s1_op1.negated is True
  assert s1_op1.absolute is True
  assert s1_op2.value == 1.5
  stmt2 = nodes[2]
  assert isinstance(stmt2, NvidiaSassInstruction)
  s2_op1 = stmt2.operands[1]
  assert isinstance(s2_op1, NvidiaSassImmediate)
  assert s2_op1.value == 0x3F800000
  assert s2_op1.is_hex is True
  stmt3 = nodes[3]
  assert isinstance(stmt3, NvidiaSassInstruction)
  s3_op1 = stmt3.operands[1]
  assert isinstance(s3_op1, NvidiaSassImmediate)
  assert s3_op1.value == -16
  assert s3_op1.is_hex is True


def test_nvidia_sass_parser_missing_semicolon_and_whitespace() -> None:
  """Verify `NvidiaSassParser` handles nvidia SASS parser missing semicolon and whitespace as expected."""
  parser = NvidiaSassParser("MOV R0, R1   ")
  nodes = parser.parse().statements
  stmt0 = nodes[0]
  assert isinstance(stmt0, NvidiaSassInstruction)
  assert stmt0.opcode == "MOV"
  assert stmt0.trailing_trivia == []


def test_nvidia_sass_parser_operand_predicates() -> None:
  """Verify `NvidiaSassParser` handles nvidia SASS parser operand predicates as expected."""
  parser = NvidiaSassParser("ISETP.NE.AND P0, !P1, PT;")
  nodes = parser.parse().statements
  stmt0 = nodes[0]
  assert isinstance(stmt0, NvidiaSassInstruction)
  op0 = stmt0.operands[0]
  op1 = stmt0.operands[1]
  op2 = stmt0.operands[2]
  assert isinstance(op0, NvidiaSassLabel)
  assert op0.name == "P0"
  assert isinstance(op1, NvidiaSassPredicate)
  assert op1.name == "P1"
  assert op1.negated is True
  assert isinstance(op2, NvidiaSassRegister)
  assert op2.name == "PT"


def test_nvidia_sass_parser_malformed_memory_and_infinite_loop_prevention() -> None:
  """Verify nvidia SASS parser malformed memory and infinite loop prevention works as expected."""
  with pytest.raises(ValueError, match="Unexpected"):
    NvidiaSassParser("LDG [R1 junk];\nMOV R0, c[0x0 junk];\nMOV R1, c[0x0][0x4 junk];\nMOV R2, .;\n").parse()


def test_nvidia_sass_parser_empty_and_semicolon() -> None:
  """Verify `parse` raises `ValueError` during nvidia SASS parser empty and semicolon."""
  assert NvidiaSassParser("").parse().statements == []
  assert NvidiaSassParser(";").parse().statements == []
  assert NvidiaSassParser("   \n ").parse().statements == []
  stmt = NvidiaSassParser("c").parse().statements[0]
  assert isinstance(stmt, NvidiaSassInstruction)
  assert stmt.opcode == "c"
  with pytest.raises(ValueError, match="Unexpected"):
    NvidiaSassParser("/").parse()


def test_nvidia_sass_parser_instruction_break_conditions() -> None:
  """Verify nvidia SASS parser instruction break conditions works as expected."""
  parser = NvidiaSassParser("MOV R0, R1 // comment here\nMOV R2, R3;")
  nodes = parser.parse().statements
  stmt0 = nodes[0]
  assert isinstance(stmt0, NvidiaSassInstruction)
  assert stmt0.opcode == "MOV"
  assert len(stmt0.operands) == 2
  with pytest.raises(ValueError, match="Unexpected"):
    NvidiaSassParser("MOV R0, \n").parse()


def test_nvidia_sass_parser_missing_lines() -> None:
  """Verify `NvidiaSassParser` returns `1` for nvidia SASS parser missing lines."""
  parser = NvidiaSassParser("LDG.E R0, [R1 - 0x10];")
  assert len(parser.parse().statements) == 1

  parser = NvidiaSassParser("@!R0 NOP;\n@R1 NOP;\n@P0 NOP;")
  assert len(parser.parse().statements) == 3

  parser = NvidiaSassParser("MOV R0, -0x1A;")
  stmts = parser.parse().statements
  assert len(stmts) == 1


def test_nvidia_sass_parser_missing_lines_2() -> None:
  """Verify `NvidiaSassParser` raises `ValueError` during nvidia SASS parser missing lines 2."""
  parser = NvidiaSassParser("@!R1 NOP;\n@P0 NOP;\n@R2 NOP;")
  assert len(parser.parse().statements) == 3

  parser = NvidiaSassParser("MOV R0, - 0x1A;")
  with pytest.raises(ValueError, match="Unexpected"):
    parser.parse()


def test_nvidia_sass_parser_missing_lines_3() -> None:
  """Verify `NvidiaSassParser` handles nvidia SASS parser missing lines 3 as expected."""
  parser = NvidiaSassParser("@P0 NOP;\n@!P1 NOP;")
  assert len(parser.parse().statements) == 2

  parser = NvidiaSassParser("MOV R0, !R1;")
  assert len(parser.parse().statements) == 1


def test_nvidia_sass_parser_predicate_variants() -> None:
  """Verify `NvidiaSassParser` handles nvidia SASS parser predicate variants as expected."""
  parser = NvidiaSassParser("@PT NOP;")
  nodes = parser.parse().statements
  stmt0 = nodes[0]
  assert isinstance(stmt0, NvidiaSassInstruction)
  assert stmt0.predicate is not None
  assert stmt0.predicate.name == "PT"

  parser = NvidiaSassParser("@P0 NOP;")
  nodes = parser.parse().statements
  stmt1 = nodes[0]
  assert isinstance(stmt1, NvidiaSassInstruction)
  assert stmt1.predicate is not None
  assert stmt1.predicate.name == "P0"

  parser = NvidiaSassParser("NOP !R0;")
  nodes = parser.parse().statements
  stmt2 = nodes[0]
  assert isinstance(stmt2, NvidiaSassInstruction)
  op0 = stmt2.operands[0]
  assert isinstance(op0, NvidiaSassPredicate)
  assert op0.negated is True


def test_nvidia_sass_parser_missing_lines_4() -> None:
  """Verify `NvidiaSassParser` handles nvidia SASS parser missing lines 4 as expected."""
  parser = NvidiaSassParser("NOP !P0;")
  nodes = parser.parse().statements
  stmt0 = nodes[0]
  assert isinstance(stmt0, NvidiaSassInstruction)
  op0 = stmt0.operands[0]
  assert isinstance(op0, NvidiaSassPredicate)
  assert op0.negated is True

  parser = NvidiaSassParser("NOP !R0;")
  nodes = parser.parse().statements
  stmt1 = nodes[0]
  assert isinstance(stmt1, NvidiaSassInstruction)
  op1 = stmt1.operands[0]
  assert isinstance(op1, NvidiaSassPredicate)
  assert op1.negated is True

  parser = NvidiaSassParser("@!R1 NOP;")
  nodes = parser.parse().statements
  stmt2 = nodes[0]
  assert isinstance(stmt2, NvidiaSassInstruction)
  assert stmt2.predicate is not None
  assert stmt2.predicate.negated is True

  parser = NvidiaSassParser("@R2 NOP;")
  nodes = parser.parse().statements
  stmt3 = nodes[0]
  assert isinstance(stmt3, NvidiaSassInstruction)
  assert stmt3.predicate is not None
  assert stmt3.predicate.negated is False

  parser = NvidiaSassParser("@P0 NOP;")
  nodes = parser.parse().statements
  stmt4 = nodes[0]
  assert isinstance(stmt4, NvidiaSassInstruction)
  assert stmt4.predicate is not None
  assert stmt4.predicate.negated is False

  parser = NvidiaSassParser("NOP P0;")
  nodes = parser.parse().statements
  stmt5 = nodes[0]
  assert isinstance(stmt5, NvidiaSassInstruction)
  op5 = stmt5.operands[0]
  assert isinstance(op5, NvidiaSassLabel)
  assert op5.name == "P0"


def test_nvidia_sass_parser_missing_lines_5() -> None:
  """Verify `NvidiaSassParser` handles nvidia SASS parser missing lines 5 as expected."""
  parser = NvidiaSassParser("@!P0_custom NOP;")
  assert len(parser.parse().statements) == 1

  parser = NvidiaSassParser("@P0_custom NOP;")
  assert len(parser.parse().statements) == 1

  parser = NvidiaSassParser("NOP !P0_custom;")
  assert len(parser.parse().statements) == 1

  parser = NvidiaSassParser("@!R0 NOP;")
  assert len(parser.parse().statements) == 1

  parser = NvidiaSassParser("@R0 NOP;")
  assert len(parser.parse().statements) == 1

  parser = NvidiaSassParser("NOP !R0;")
  assert len(parser.parse().statements) == 1


def test_nvidia_sass_parser_directive_params_edge() -> None:
  """Verify `NvidiaSassParser` handles nvidia SASS parser directive params edge as expected."""
  parser = NvidiaSassParser(".reqntid 1")
  stmts = parser.parse().statements
  assert len(stmts) == 1
  stmt0 = stmts[0]
  assert isinstance(stmt0, NvidiaSassDirective)
  assert stmt0.params == ["1"]
