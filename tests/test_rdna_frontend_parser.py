"""Test module."""

import pytest
from ml_switcheroo.core.compiler.frontends.rdna.parser import RdnaParser
from ml_switcheroo.core.compiler.frontends.rdna.cst import (
  RdnaComment,
  RdnaDirective,
  RdnaLabel,
  RdnaInstruction,
  RdnaSGPR,
  RdnaVGPR,
  RdnaModifier,
  RdnaLabelRef,
  RdnaMemory,
)


def test_parser_empty():
  """Test element."""
  parser = RdnaParser("")
  mod = parser.parse()
  assert len(mod.statements) == 0

  parser = RdnaParser("   \n  ")
  mod = parser.parse()
  assert len(mod.statements) == 0


def test_parser_comments():
  """Test element."""
  parser = RdnaParser("; just a comment\n  ; another comment")
  mod = parser.parse()
  assert len(mod.statements) == 2
  assert isinstance(mod.statements[0], RdnaComment)
  assert mod.statements[0].text == " just a comment"
  assert mod.statements[1].text == " another comment"


def test_parser_directive():
  """Test element."""
  parser = RdnaParser(".text\n.globl main\n.type main, function")
  mod = parser.parse()
  assert len(mod.statements) == 3
  assert isinstance(mod.statements[0], RdnaDirective)
  assert mod.statements[0].name == "text"
  assert mod.statements[1].name == "globl"
  assert mod.statements[1].params == ["main"]
  assert mod.statements[2].name == "type"
  assert mod.statements[2].params == ["main", "function"]


def test_parser_label():
  """Test element."""
  parser = RdnaParser("main:\nL1:")
  mod = parser.parse()
  assert len(mod.statements) == 2
  assert isinstance(mod.statements[0], RdnaLabel)
  assert mod.statements[0].name == "main"
  assert mod.statements[1].name == "L1"


def test_parser_instruction_no_args():
  """Test element."""
  parser = RdnaParser("s_endpgm")
  mod = parser.parse()
  assert len(mod.statements) == 1
  assert isinstance(mod.statements[0], RdnaInstruction)
  assert mod.statements[0].opcode == "s_endpgm"
  assert len(mod.statements[0].operands) == 0


def test_parser_instruction_registers():
  """Test element."""
  parser = RdnaParser("v_add_f32 v0, s[1:2], v3")
  mod = parser.parse()
  inst = mod.statements[0]
  assert inst.opcode == "v_add_f32"
  assert len(inst.operands) == 3

  op0 = inst.operands[0]
  assert isinstance(op0, RdnaVGPR)
  assert op0.index == 0
  assert op0.count == 1

  op1 = inst.operands[1]
  assert isinstance(op1, RdnaSGPR)
  assert op1.index == 1
  assert op1.count == 2

  op2 = inst.operands[2]
  assert isinstance(op2, RdnaVGPR)
  assert op2.index == 3


def test_parser_immediates():
  """Test element."""
  parser = RdnaParser("v_mov_b32 v0, 42\nv_mov_b32 v1, -42\nv_mov_b32 v2, +42")
  mod = parser.parse()
  assert mod.statements[0].operands[1].value == 42
  assert mod.statements[1].operands[1].value == -42
  assert mod.statements[2].operands[1].value == 42

  parser = RdnaParser("v_mov_b32 v0, 0x2a\nv_mov_b32 v1, -0x2a\nv_mov_b32 v2, +0x2a")
  mod = parser.parse()
  assert mod.statements[0].operands[1].value == 42
  assert mod.statements[1].operands[1].value == -42
  assert mod.statements[2].operands[1].value == 42


def test_parser_immediates_float():
  """Test element."""
  parser = RdnaParser("v_mov_b32 v0, 3.14\nv_mov_b32 v1, -3.14\nv_mov_b32 v2, +3.14")
  mod = parser.parse()
  assert mod.statements[0].operands[1].value == 3.14
  assert mod.statements[1].operands[1].value == -3.14
  assert mod.statements[2].operands[1].value == 3.14


def test_parser_memory():
  """Test element."""
  # Correct memory syntax: `[s[2:3]]`, `[s[2:3] + 4]`, `[s[2:3] - 4]`
  parser = RdnaParser("s_load_dword s0, [s[2:3]]\ns_load_dword s0, [s[2:3] + 4]\ns_load_dword s0, [s[2:3] - 8]")
  mod = parser.parse()

  mem1 = mod.statements[0].operands[1]
  assert isinstance(mem1, RdnaMemory)
  assert mem1.offset == 0

  mem2 = mod.statements[1].operands[1]
  assert isinstance(mem2, RdnaMemory)
  assert mem2.offset == 4

  mem3 = mod.statements[2].operands[1]
  assert isinstance(mem3, RdnaMemory)
  assert mem3.offset == -8


def test_parser_modifier():
  """Test element."""
  parser = RdnaParser("v_add_f32 v0, v1, v2, glc\n s_branch L1")
  mod = parser.parse()
  inst = mod.statements[0]
  assert isinstance(inst.operands[-1], RdnaModifier)
  assert inst.operands[-1].name == "glc"

  inst2 = mod.statements[1]
  assert isinstance(inst2.operands[0], RdnaLabelRef)
  assert inst2.operands[0].name == "L1"


def test_parser_invalid():
  """Test element."""
  parser = RdnaParser("v_add_f32 @@@")
  with pytest.raises(ValueError):
    parser.parse()


def test_parser_lexer_mismatch():
  """Test element."""
  parser = RdnaParser("!")
  with pytest.raises(ValueError, match="Unexpected '!'"):
    parser.parse()


def test_parser_eof_trivia():
  """Test element."""
  parser = RdnaParser("v_add_f32 v0, v1 ")
  mod = parser.parse()
  assert mod.statements[-1].trailing_trivia[-1].text == " "
