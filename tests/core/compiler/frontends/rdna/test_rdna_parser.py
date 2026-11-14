"""Test suite for the RDNA Parser Extra module."""

import pytest

from ml_switcheroo.core.compiler.frontends.rdna.cst import (
  RdnaComment,
  RdnaDirective,
  RdnaImmediate,
  RdnaInstruction,
  RdnaLabel,
  RdnaModifier,
  RdnaNode,
  RdnaSGPR,
  RdnaVGPR,
)
from ml_switcheroo.core.compiler.frontends.rdna.parser import RdnaParser


def test_rdna_parser_missing() -> None:
  """Verify `RdnaParser` returns `2` for RDNA parser missing."""
  parser = RdnaParser(".text\n.global main")
  assert len(parser.parse().statements) == 2
  parser = RdnaParser("v_add_f32 v0, v1, v2\n.text")
  assert len(parser.parse().statements) == 2


def test_rdna_parser_comments_and_empty() -> None:
  """Verify `RdnaParser` returns `2` for RDNA parser comments and empty."""
  parser = RdnaParser("; comment\n  \n;  \n")
  nodes: list[RdnaNode] = parser.parse().statements
  assert len(nodes) == 2
  stmt0 = nodes[0]
  assert isinstance(stmt0, RdnaComment)
  assert stmt0.text == " comment"


def test_rdna_parser_directive_params() -> None:
  """Verify `RdnaParser` handles RDNA parser directive params as expected."""
  parser = RdnaParser(".reqntid 128, 1, 1\n")
  nodes: list[RdnaNode] = parser.parse().statements
  stmt0 = nodes[0]
  assert isinstance(stmt0, RdnaDirective)
  assert stmt0.name == "reqntid"
  assert stmt0.params == ["128", "1", "1"]


def test_rdna_parser_memory_and_labels() -> None:
  """Verify `RdnaParser` handles RDNA parser memory and labels as expected."""
  parser = RdnaParser("label_target:\nv_add_f32 v0, v1, v2")
  nodes: list[RdnaNode] = parser.parse().statements
  stmt0 = nodes[0]
  stmt1 = nodes[1]
  assert isinstance(stmt0, RdnaLabel)
  assert stmt0.name == "label_target"
  assert isinstance(stmt1, RdnaInstruction)
  assert stmt1.opcode == "v_add_f32"


def test_rdna_parser_registers() -> None:
  """Verify `RdnaParser` returns `0` for RDNA parser registers."""
  parser = RdnaParser("s_mov_b32 s0, s[1:2]\nv_mov_b32 v0, v[1:2]")
  nodes: list[RdnaNode] = parser.parse().statements
  stmt0 = nodes[0]
  assert isinstance(stmt0, RdnaInstruction)
  s0_op0 = stmt0.operands[0]
  s0_op1 = stmt0.operands[1]
  assert isinstance(s0_op0, RdnaSGPR)
  assert isinstance(s0_op1, RdnaSGPR)
  assert s0_op0.index == 0
  assert s0_op1.index == 1
  assert s0_op1.count == 2
  stmt1 = nodes[1]
  assert isinstance(stmt1, RdnaInstruction)
  s1_op0 = stmt1.operands[0]
  s1_op1 = stmt1.operands[1]
  assert isinstance(s1_op0, RdnaVGPR)
  assert isinstance(s1_op1, RdnaVGPR)
  assert s1_op0.index == 0
  assert s1_op1.index == 1
  assert s1_op1.count == 2


def test_rdna_parser_immediates_and_modifiers() -> None:
  """Verify `RdnaParser` handles RDNA parser immediates and modifiers as expected."""
  parser = RdnaParser("v_add_f32 v0, -1.5, -0x10\nglobal_load_dword v0, v[1:2], off, offset:0x10")
  nodes: list[RdnaNode] = parser.parse().statements
  stmt0 = nodes[0]
  assert isinstance(stmt0, RdnaInstruction)
  s0_op1 = stmt0.operands[1]
  s0_op2 = stmt0.operands[2]
  assert isinstance(s0_op1, RdnaImmediate)
  assert isinstance(s0_op2, RdnaImmediate)
  assert s0_op1.value == -1.5
  assert s0_op2.value == -16
  assert s0_op2.is_hex is True
  stmt1 = nodes[1]
  assert isinstance(stmt1, RdnaInstruction)
  s1_op2 = stmt1.operands[2]
  s1_op3 = stmt1.operands[3]
  assert isinstance(s1_op2, RdnaModifier)
  assert isinstance(s1_op3, RdnaModifier)
  assert s1_op2.name == "off"
  assert s1_op3.name == "offset:0x10"


def test_rdna_parser_infinite_loop_prevention() -> None:
  """Verify `parse` raises `ValueError` during RDNA parser infinite loop prevention."""
  with pytest.raises(ValueError, match="Unexpected"):
    RdnaParser("v_add_f32 v0, - \n").parse()


def test_rdna_parser_empty_and_semicolon() -> None:
  """Verify `parse` raises `ValueError` during RDNA parser empty and semicolon."""
  assert RdnaParser("").parse().statements == []
  assert RdnaParser("   \n ").parse().statements == []
  stmt = RdnaParser("c").parse().statements[0]
  assert isinstance(stmt, RdnaInstruction)
  assert stmt.opcode == "c"
  with pytest.raises(ValueError, match="Unexpected"):
    RdnaParser("/").parse()


def test_rdna_parser_instruction_break_conditions() -> None:
  """Verify `RdnaParser` raises `ValueError` during RDNA parser instruction break conditions."""
  parser = RdnaParser("v_add_f32 v0, v1 ; comment here\n")
  nodes: list[RdnaNode] = parser.parse().statements
  stmt0 = nodes[0]
  assert isinstance(stmt0, RdnaInstruction)
  assert stmt0.opcode == "v_add_f32"
  assert len(stmt0.operands) == 2
  with pytest.raises(ValueError, match="Unexpected"):
    RdnaParser("v_add_f32 v0, \n").parse()


def test_rdna_parser_missing_lines() -> None:
  """Verify `parse` handles RDNA parser missing lines as expected."""
  stmt = RdnaParser("unknown_token").parse().statements[0]
  assert isinstance(stmt, RdnaInstruction)
  assert stmt.opcode == "unknown_token"


def test_rdna_parser_missing_lines_2() -> None:
  """Verify `RdnaParser` returns `1` for RDNA parser missing lines 2."""
  parser = RdnaParser("v_add_f32 v0, [s0], [s0+4], [s0-4]")
  assert len(parser.parse().statements) == 1

  parser = RdnaParser("v_add_f32 10, 0x1A, some_label, mod_name:0")
  assert len(parser.parse().statements) == 1

  parser = RdnaParser("unknown_opcode")
  assert len(parser.parse().statements) == 1

  parser = RdnaParser(".macro")
  assert len(parser.parse().statements) == 1


def test_rdna_parser_missing_lines_3() -> None:
  """Verify `RdnaParser` returns `1` for RDNA parser missing lines 3."""
  parser = RdnaParser("v_add_f32 +10, +0x1A")
  assert len(parser.parse().statements) == 1

  parser = RdnaParser(".directive param1, param2")
  stmts = parser.parse().statements
  assert len(stmts) == 1
  stmt0 = stmts[0]
  assert isinstance(stmt0, RdnaDirective)
  assert stmt0.params == ["param1", "param2"]


def test_rdna_parser_missing_lines_4() -> None:
  """Verify `RdnaParser` returns `1` for RDNA parser missing lines 4."""
  parser = RdnaParser(".directive param1")
  assert len(parser.parse().statements) == 1

  parser = RdnaParser(".directive 1+1")
  assert len(parser.parse().statements) == 1
