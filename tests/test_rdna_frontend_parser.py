"""Test module."""

from typing import List

import pytest
from lark import Token

from ml_switcheroo.core.compiler.frontends.rdna.cst import (
  RdnaComment,
  RdnaDirective,
  RdnaInstruction,
  RdnaLabel,
  RdnaLabelRef,
  RdnaMemory,
  RdnaModifier,
  RdnaModule,
  RdnaNode,
  RdnaSGPR,
  RdnaVGPR,
)
from ml_switcheroo.core.compiler.frontends.rdna.parser import RdnaParser, RdnaTransformer


def test_parser_empty() -> None:
  """Docstring."""
  parser: RdnaParser = RdnaParser("")
  mod: RdnaModule = parser.parse()
  assert len(mod.statements) == 0

  parser2: RdnaParser = RdnaParser("   \n  ")
  mod2: RdnaModule = parser2.parse()
  assert len(mod2.statements) == 0


def test_parser_comments() -> None:
  """Docstring."""
  parser: RdnaParser = RdnaParser("; just a comment\n  ; another comment")
  mod: RdnaModule = parser.parse()
  assert len(mod.statements) == 2
  assert isinstance(mod.statements[0], RdnaComment)
  assert mod.statements[0].text == " just a comment"
  assert getattr(mod.statements[1], "text", "") == " another comment"


def test_parser_directive() -> None:
  """Docstring."""
  parser: RdnaParser = RdnaParser(".text\n.globl main\n.type main, function")
  mod: RdnaModule = parser.parse()
  assert len(mod.statements) == 3
  assert isinstance(mod.statements[0], RdnaDirective)
  assert mod.statements[0].name == "text"
  assert getattr(mod.statements[1], "name", "") == "globl"
  assert getattr(mod.statements[1], "params", []) == ["main"]
  assert getattr(mod.statements[2], "name", "") == "type"
  assert getattr(mod.statements[2], "params", []) == ["main", "function"]


def test_parser_label() -> None:
  """Docstring."""
  parser: RdnaParser = RdnaParser("main:\nL1:")
  mod: RdnaModule = parser.parse()
  assert len(mod.statements) == 2
  assert isinstance(mod.statements[0], RdnaLabel)
  assert mod.statements[0].name == "main"
  assert getattr(mod.statements[1], "name", "") == "L1"


def test_parser_instruction_no_args() -> None:
  """Docstring."""
  parser: RdnaParser = RdnaParser("s_endpgm")
  mod: RdnaModule = parser.parse()
  assert len(mod.statements) == 1
  assert isinstance(mod.statements[0], RdnaInstruction)
  assert mod.statements[0].opcode == "s_endpgm"
  assert len(mod.statements[0].operands) == 0


def test_parser_instruction_registers() -> None:
  """Docstring."""
  parser: RdnaParser = RdnaParser("v_add_f32 v0, s[1:2], v3")
  mod: RdnaModule = parser.parse()
  inst: RdnaInstruction = getattr(mod, "statements")[0]
  assert inst.opcode == "v_add_f32"
  assert len(inst.operands) == 3

  op0: RdnaNode = inst.operands[0]
  assert isinstance(op0, RdnaVGPR)
  assert op0.index == 0
  assert getattr(op0, "count", 0) == 1

  op1: RdnaNode = inst.operands[1]
  assert isinstance(op1, RdnaSGPR)
  assert op1.index == 1
  assert getattr(op1, "count", 0) == 2

  op2: RdnaNode = inst.operands[2]
  assert isinstance(op2, RdnaVGPR)
  assert op2.index == 3


def test_parser_immediates() -> None:
  """Docstring."""
  parser: RdnaParser = RdnaParser("v_mov_b32 v0, 42\nv_mov_b32 v1, -42\nv_mov_b32 v2, +42")
  mod: RdnaModule = parser.parse()
  assert getattr(getattr(mod.statements[0], "operands", [])[1], "value", None) == 42
  assert getattr(getattr(mod.statements[1], "operands", [])[1], "value", None) == -42
  assert getattr(getattr(mod.statements[2], "operands", [])[1], "value", None) == 42

  parser2: RdnaParser = RdnaParser("v_mov_b32 v0, 0x2a\nv_mov_b32 v1, -0x2a\nv_mov_b32 v2, +0x2a")
  mod2: RdnaModule = parser2.parse()
  assert getattr(getattr(mod2.statements[0], "operands", [])[1], "value", None) == 42
  assert getattr(getattr(mod2.statements[1], "operands", [])[1], "value", None) == -42
  assert getattr(getattr(mod2.statements[2], "operands", [])[1], "value", None) == 42


def test_parser_immediates_float() -> None:
  """Docstring."""
  parser: RdnaParser = RdnaParser("v_mov_b32 v0, 3.14\nv_mov_b32 v1, -3.14\nv_mov_b32 v2, +3.14")
  mod: RdnaModule = parser.parse()
  assert getattr(getattr(mod.statements[0], "operands", [])[1], "value", None) == 3.14
  assert getattr(getattr(mod.statements[1], "operands", [])[1], "value", None) == -3.14
  assert getattr(getattr(mod.statements[2], "operands", [])[1], "value", None) == 3.14


def test_parser_memory() -> None:
  """Docstring."""
  # Correct memory syntax: `[s[2:3]]`, `[s[2:3] + 4]`, `[s[2:3] - 4]`
  parser: RdnaParser = RdnaParser(
    "s_load_dword s0, [s[2:3]]\ns_load_dword s0, [s[2:3] + 4]\ns_load_dword s0, [s[2:3] - 8]"
  )
  mod: RdnaModule = parser.parse()

  mem1: RdnaNode = getattr(mod.statements[0], "operands")[1]
  assert isinstance(mem1, RdnaMemory)
  assert mem1.offset == 0

  mem2: RdnaNode = getattr(mod.statements[1], "operands")[1]
  assert isinstance(mem2, RdnaMemory)
  assert mem2.offset == 4

  mem3: RdnaNode = getattr(mod.statements[2], "operands")[1]
  assert isinstance(mem3, RdnaMemory)
  assert mem3.offset == -8


def test_parser_modifier() -> None:
  """Docstring."""
  parser: RdnaParser = RdnaParser("v_add_f32 v0, v1, v2, glc\n s_branch L1")
  mod: RdnaModule = parser.parse()
  inst: RdnaInstruction = getattr(mod, "statements")[0]
  assert isinstance(inst.operands[-1], RdnaModifier)
  assert inst.operands[-1].name == "glc"

  inst2: RdnaInstruction = getattr(mod, "statements")[1]
  assert isinstance(inst2.operands[0], RdnaLabelRef)
  assert inst2.operands[0].name == "L1"


def test_parser_invalid() -> None:
  """Docstring."""
  parser: RdnaParser = RdnaParser("v_add_f32 @@@")
  with pytest.raises(ValueError):
    parser.parse()


def test_parser_lexer_mismatch() -> None:
  """Docstring."""
  parser: RdnaParser = RdnaParser("!")
  with pytest.raises(ValueError, match="Unexpected '!'"):
    parser.parse()


def test_parser_eof_trivia() -> None:
  """Docstring."""
  parser: RdnaParser = RdnaParser("v_add_f32 v0, v1 ")
  mod: RdnaModule = parser.parse()
  assert getattr(mod.statements[-1], "trailing_trivia")[-1].text == " "


# --- Merged from test_rdna_frontend_parser_extra.py ---


def test_parser_empty_line() -> None:
  """Docstring."""
  parser: RdnaParser = RdnaParser("  \n  v_add_f32 v0, v1, v2")
  parser.parse()


def test_parser_modifier_extra() -> None:
  """Docstring."""
  transformer: RdnaTransformer = RdnaTransformer()
  res: RdnaModifier = transformer.modifier([Token("MODIFIER", "row_mask:0xf")])
  assert res.name == "row_mask:0xf"


def test_parser_eof_trivia_extra() -> None:
  """Docstring."""
  parser: RdnaParser = RdnaParser("v_add_f32 v0, v1, v2 ; eof comment")
  parser.parse()


def test_param_children() -> None:
  """Docstring."""
  from ml_switcheroo.core.compiler.frontends.rdna.parser import RdnaTransformer

  transformer: RdnaTransformer = RdnaTransformer()

  class DummyToken:
    def __init__(self) -> None:
      self.children: List[Token] = [Token("A", "b"), Token("B", "c")]

  # We call directive directly
  # children = [ DOT, Token("IDENTIFIER", "name"), param_list ]
  # param_list is a list of parameters
  res: RdnaDirective = transformer.directive([Token("DOT", "."), Token("IDENTIFIER", "my_dir"), [DummyToken()]])
  assert res.name == "my_dir"
  assert "bc" in getattr(res, "params")[0]
