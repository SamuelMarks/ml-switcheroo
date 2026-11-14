"""Test suite for rdna frontend parser."""

import pytest
from lark import Token

from ml_switcheroo.core.compiler.frontends.rdna.cst import (
  RdnaComment,
  RdnaDirective,
  RdnaImmediate,
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
from ml_switcheroo.core.compiler.frontends.rdna.parser import RdnaLexer, RdnaParser, RdnaTransformer


def test_parser_empty() -> None:
  """Verify `RdnaParser` returns `0` for parser empty."""
  parser: RdnaParser = RdnaParser("")
  mod: RdnaModule = parser.parse()
  assert len(mod.statements) == 0

  parser2: RdnaParser = RdnaParser("   \n  ")
  mod2: RdnaModule = parser2.parse()
  assert len(mod2.statements) == 0


def test_parser_comments() -> None:
  """Verify `RdnaParser` returns `2` for parser comments."""
  parser: RdnaParser = RdnaParser("; just a comment\n  ; another comment")
  mod: RdnaModule = parser.parse()
  assert len(mod.statements) == 2
  assert isinstance(mod.statements[0], RdnaComment)
  assert mod.statements[0].text == " just a comment"
  stmt1 = mod.statements[1]
  assert isinstance(stmt1, RdnaComment)
  assert stmt1.text == " another comment"


def test_parser_directive() -> None:
  """Verify `RdnaParser` returns `3` for parser directive."""
  parser: RdnaParser = RdnaParser(".text\n.globl main\n.type main, function")
  mod: RdnaModule = parser.parse()
  assert len(mod.statements) == 3
  assert isinstance(mod.statements[0], RdnaDirective)
  assert mod.statements[0].name == "text"
  stmt1 = mod.statements[1]
  stmt2 = mod.statements[2]
  assert isinstance(stmt1, RdnaDirective)
  assert isinstance(stmt2, RdnaDirective)
  assert stmt1.name == "globl"
  assert stmt1.params == ["main"]
  assert stmt2.name == "type"
  assert stmt2.params == ["main", "function"]


def test_parser_label() -> None:
  """Verify `RdnaParser` returns `2` for parser label."""
  parser: RdnaParser = RdnaParser("main:\nL1:")
  mod: RdnaModule = parser.parse()
  assert len(mod.statements) == 2
  assert isinstance(mod.statements[0], RdnaLabel)
  assert mod.statements[0].name == "main"
  stmt1 = mod.statements[1]
  assert isinstance(stmt1, RdnaLabel)
  assert stmt1.name == "L1"


def test_parser_instruction_no_args() -> None:
  """Verify `RdnaParser` returns `1` for parser instruction no args."""
  parser: RdnaParser = RdnaParser("s_endpgm")
  mod: RdnaModule = parser.parse()
  assert len(mod.statements) == 1
  assert isinstance(mod.statements[0], RdnaInstruction)
  assert mod.statements[0].opcode == "s_endpgm"
  assert len(mod.statements[0].operands) == 0


def test_parser_instruction_registers() -> None:
  """Verify `RdnaParser` returns `3` for parser instruction registers."""
  parser: RdnaParser = RdnaParser("v_add_f32 v0, s[1:2], v3")
  mod: RdnaModule = parser.parse()
  stmt0 = mod.statements[0]
  assert isinstance(stmt0, RdnaInstruction)
  inst: RdnaInstruction = stmt0
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


def _extract_imm_val(stmt: RdnaNode) -> int | float:
  """Extract immediate value from second operand of an RDNA instruction.

  Args:
      stmt: Parsed RDNA statement node.

  Returns:
      Numeric immediate value from the instruction's second operand.
  """
  assert isinstance(stmt, RdnaInstruction)
  imm = stmt.operands[1]
  assert isinstance(imm, RdnaImmediate)
  return imm.value


def test_parser_immediates() -> None:
  """Verify `RdnaParser` returns `42` for parser immediates."""
  parser: RdnaParser = RdnaParser("v_mov_b32 v0, 42\nv_mov_b32 v1, -42\nv_mov_b32 v2, +42")
  mod: RdnaModule = parser.parse()
  assert _extract_imm_val(mod.statements[0]) == 42
  assert _extract_imm_val(mod.statements[1]) == -42
  assert _extract_imm_val(mod.statements[2]) == 42

  parser2: RdnaParser = RdnaParser("v_mov_b32 v0, 0x2a\nv_mov_b32 v1, -0x2a\nv_mov_b32 v2, +0x2a")
  mod2: RdnaModule = parser2.parse()
  assert _extract_imm_val(mod2.statements[0]) == 42
  assert _extract_imm_val(mod2.statements[1]) == -42
  assert _extract_imm_val(mod2.statements[2]) == 42


def test_parser_immediates_float() -> None:
  """Verify `RdnaParser` handles parser immediates float as expected."""
  parser: RdnaParser = RdnaParser("v_mov_b32 v0, 3.14\nv_mov_b32 v1, -3.14\nv_mov_b32 v2, +3.14")
  mod: RdnaModule = parser.parse()
  assert _extract_imm_val(mod.statements[0]) == 3.14
  assert _extract_imm_val(mod.statements[1]) == -3.14
  assert _extract_imm_val(mod.statements[2]) == 3.14


def test_parser_memory() -> None:
  """Verify `RdnaParser` returns `0` for parser memory."""
  # Correct memory syntax: `[s[2:3]]`, `[s[2:3] + 4]`, `[s[2:3] - 4]`
  parser: RdnaParser = RdnaParser(
    "s_load_dword s0, [s[2:3]]\ns_load_dword s0, [s[2:3] + 4]\ns_load_dword s0, [s[2:3] - 8]"
  )
  mod: RdnaModule = parser.parse()

  s0, s1, s2 = mod.statements[0], mod.statements[1], mod.statements[2]
  assert isinstance(s0, RdnaInstruction)
  assert isinstance(s1, RdnaInstruction)
  assert isinstance(s2, RdnaInstruction)

  mem1 = s0.operands[1]
  assert isinstance(mem1, RdnaMemory)
  assert mem1.offset == 0

  mem2 = s1.operands[1]
  assert isinstance(mem2, RdnaMemory)
  assert mem2.offset == 4

  mem3 = s2.operands[1]
  assert isinstance(mem3, RdnaMemory)
  assert mem3.offset == -8


def test_parser_modifier() -> None:
  """Verify `RdnaParser` handles parser modifier as expected."""
  parser: RdnaParser = RdnaParser("v_add_f32 v0, v1, v2, glc\n s_branch L1")
  mod: RdnaModule = parser.parse()
  stmt0 = mod.statements[0]
  assert isinstance(stmt0, RdnaInstruction)
  inst: RdnaInstruction = stmt0
  assert isinstance(inst.operands[-1], RdnaModifier)
  assert inst.operands[-1].name == "glc"

  stmt1 = mod.statements[1]
  assert isinstance(stmt1, RdnaInstruction)
  inst2: RdnaInstruction = stmt1
  assert isinstance(inst2.operands[0], RdnaLabelRef)
  assert inst2.operands[0].name == "L1"


def test_parser_invalid() -> None:
  """Verify `RdnaParser` raises `ValueError` during parser invalid."""
  parser: RdnaParser = RdnaParser("v_add_f32 @@@")
  with pytest.raises(ValueError, match="Unexpected"):
    parser.parse()


def test_parser_lexer_mismatch() -> None:
  """Verify `RdnaParser` raises `ValueError` during parser lexer mismatch."""
  parser: RdnaParser = RdnaParser("!")
  with pytest.raises(ValueError, match="Unexpected '!'"):
    parser.parse()


def test_parser_eof_trivia() -> None:
  """Verify `RdnaParser` handles parser EOF trivia as expected."""
  parser: RdnaParser = RdnaParser("v_add_f32 v0, v1 ")
  mod: RdnaModule = parser.parse()
  assert mod.statements[-1].trailing_trivia[-1].text == " "


def test_parser_empty_line() -> None:
  """Verify `RdnaParser` returns `1` for parser empty line."""
  parser: RdnaParser = RdnaParser("  \n  v_add_f32 v0, v1, v2")
  _res = parser.parse()
  assert len(_res.statements) == 1
  stmt0 = _res.statements[0]
  assert isinstance(stmt0, RdnaInstruction)
  assert stmt0.opcode == "v_add_f32"


def test_parser_modifier_extra() -> None:
  """Verify `RdnaTransformer` handles parser modifier extra as expected."""
  transformer: RdnaTransformer = RdnaTransformer()
  res: RdnaModifier = transformer.modifier([Token("MODIFIER", "row_mask:0xf")])
  assert res.name == "row_mask:0xf"


def test_parser_eof_trivia_extra() -> None:
  """Verify `RdnaParser` returns `2` for parser EOF trivia extra."""
  parser: RdnaParser = RdnaParser("v_add_f32 v0, v1, v2 ; eof comment")
  _res = parser.parse()
  assert len(_res.statements) == 2
  last_stmt = _res.statements[-1]
  assert isinstance(last_stmt, RdnaComment)
  assert "eof comment" in last_stmt.text


def test_param_children() -> None:
  """Verify `RdnaTransformer` emits `bc` for param children."""
  from ml_switcheroo.core.compiler.frontends.rdna.parser import RdnaTransformer

  transformer: RdnaTransformer = RdnaTransformer()

  class DummyToken:
    """Sentinel `DummyToken` fixture class used in RDNA frontend parser tests."""

    def __init__(self) -> None:
      """Initialize test helper instance."""
      self.children: list[Token] = [Token("A", "b"), Token("B", "c")]

  # We call directive directly
  # children = [ DOT, Token("IDENTIFIER", "name"), param_list ]
  # param_list is a list of parameters
  res: RdnaDirective = transformer.directive([Token("DOT", "."), Token("IDENTIFIER", "my_dir"), [DummyToken()]])
  assert res.name == "my_dir"
  assert "bc" in res.params[0]


def test_rdna_tokenize_lexer_state_variants() -> None:
  """Test RdnaLexer with object having text attribute and arbitrary object."""

  class ObjWithText:
    """Sentinel `ObjWithText` fixture class used in RDNA frontend parser tests."""

    text = "v_add_f32 v0, v1, v2;\n"

  lexer = RdnaLexer(None)
  tokens1 = list(lexer.lex(ObjWithText()))
  assert len(tokens1) > 0

  class ObjWithoutText:
    """Sentinel `ObjWithoutText` fixture class used in RDNA frontend parser tests."""

    def __str__(self) -> str:
      """Return string representation.

      Returns:
          Resulting `str` from str  .
      """
      return "v_add_f32 v0, v1, v2;\n"

  tokens2 = list(lexer.lex(ObjWithoutText()))
  assert len(tokens2) > 0


def test_directive_single_non_list_param() -> None:
  """Test directive transformation with a single non-list parameter."""
  transformer: RdnaTransformer = RdnaTransformer()
  directive: RdnaDirective = transformer.directive([Token("DOT", "."), Token("IDENTIFIER", "my_dir"), "single_param"])
  assert directive.params == ["single_param"]


def test_module_eof_trivia_empty_stmts_branch() -> None:
  """Test module transformation with EOF_TRIVIA and subsequent None element."""
  transformer: RdnaTransformer = RdnaTransformer()
  mod: RdnaModule = transformer.module([Token("EOF_TRIVIA", ""), None])
  assert len(mod.statements) == 0
