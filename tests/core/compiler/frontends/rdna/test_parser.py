"""Unit tests for `RdnaParser`, `RdnaTransformer`, and RDNA CST formatting (`ml_switcheroo.core.compiler.frontends.rdna`)."""

import pytest
from lark import Token

from ml_switcheroo.core.compiler.frontends.rdna.analysis import RdnaAnalyzer
from ml_switcheroo.core.compiler.frontends.rdna.cst import (
  RdnaComment,
  RdnaDirective,
  RdnaImmediate,
  RdnaInstruction,
  RdnaLabel,
  RdnaLabelRef,
  RdnaMemory,
  RdnaModifier,
  RdnaNode,
  RdnaSGPR,
  RdnaVGPR,
  c_SGPR,
  c_VGPR,
)
from ml_switcheroo.core.compiler.frontends.rdna.parser import RdnaParser, RdnaToken, RdnaTransformer, _get_trivia
from ml_switcheroo.core.cst.base import Trivia

_NL = chr(10)


def test_rdna_parser_comments_labels_directives_and_instructions() -> None:
  """Verify `RdnaParser` parses comments, labels, directives, registers, memory, immediates, and modifiers."""
  assert RdnaParser("").parse().statements == []
  empty_triv = RdnaParser(f"  {_NL}").parse()
  assert empty_triv.statements == []
  assert empty_triv.leading_trivia == []
  assert len(RdnaParser(f"v_mov_b32 v0, v1{_NL}{_NL}  {_NL}").parse().statements) == 1

  triv_mod = RdnaParser(f"  ; hello world{_NL}  v_nop{_NL}").parse()
  c0 = triv_mod.statements[0]
  assert isinstance(c0, RdnaComment)
  assert c0.text == " hello world"
  assert c0.leading_trivia[0].text == "  "

  lbl = RdnaParser("main:").parse().statements[0]
  assert isinstance(lbl, RdnaLabel)
  assert lbl.name == "main"

  d0 = RdnaParser(".text").parse().statements[0]
  assert isinstance(d0, RdnaDirective)
  assert d0.name == "text"
  assert d0.params == []

  d_params = RdnaParser('.global main, 5, 0x1, -5, -0x1, +5, +0x1, off, "str", bar:baz').parse().statements[0]
  assert isinstance(d_params, RdnaDirective)
  assert len(d_params.params) == 10
  assert len(RdnaParser(f".global_base 1 {_NL} .global_base 2").parse().statements) == 2
  assert len(RdnaParser(f"v_add {_NL} v_sub").parse().statements) == 2

  nop = RdnaParser("s_waitcnt").parse().statements[0]
  assert isinstance(nop, RdnaInstruction)
  assert nop.opcode == "s_waitcnt"
  assert nop.operands == []

  # Registers (ranges and singular) + special label ref (`exec`)
  reg_inst = RdnaParser("v_add_f32 v[0:1], s[0:1], v10, s20, exec, my_label").parse().statements[0]
  assert isinstance(reg_inst, RdnaInstruction)
  op0, op1, op2, op3, op4, op5 = reg_inst.operands
  assert isinstance(op0, RdnaVGPR)
  assert (op0.index, op0.count) == (0, 2)
  assert isinstance(op1, RdnaSGPR)
  assert (op1.index, op1.count) == (0, 2)
  assert isinstance(op2, RdnaVGPR)
  assert (op2.index, op2.count) == (10, 1)
  assert isinstance(op3, RdnaSGPR)
  assert (op3.index, op3.count) == (20, 1)
  assert isinstance(op4, RdnaLabelRef)
  assert op4.name == "exec"
  assert isinstance(op5, RdnaLabelRef)
  assert op5.name == "my_label"

  # Memory operands: zero, +dec, +hex, -dec, -hex
  for asm_mem, expected_off in (
    ("s_load_dword s0, [s[0:1]]", 0),
    ("v_add [v0 + 4]", 4),
    ("s_load_dword s0, [s[0:1] + 0x4]", 4),
    ("v_add [v0 - 10]", -10),
    ("s_load_dword s0, [s[0:1] - 0x4]", -4),
  ):
    m_inst = RdnaParser(asm_mem).parse().statements[0]
    assert isinstance(m_inst, RdnaInstruction)
    mem_op = m_inst.operands[-1]
    assert isinstance(mem_op, RdnaMemory)
    assert mem_op.offset == expected_off

  # Immediates: int, float, hex, negative, positive
  for lit_str, expected_val, expected_hex in (
    ("5", 5, False),
    ("5.0", 5.0, False),
    ("0x5", 5, True),
    ("-5", -5, False),
    ("-5.0", -5.0, False),
    ("-0x5", -5, True),
    ("+5", 5, False),
    ("+5.0", 5.0, False),
    ("+0x5", 5, True),
  ):
    i_inst = RdnaParser(f"v_mov_b32 v0, {lit_str}").parse().statements[0]
    assert isinstance(i_inst, RdnaInstruction)
    imm_op = i_inst.operands[1]
    assert isinstance(imm_op, RdnaImmediate)
    assert imm_op.value == expected_val
    assert imm_op.is_hex is expected_hex

  # Modifiers
  for mod_asm, expected_mod in (
    ("s_waitcnt vmcnt(0)", "vmcnt(0)"),
    ("v_add_f32 v0, v1, off", "off"),
    ("v_add_f32 v0, glc", "glc"),
    ("v_add_f32 v0, row_mask:0xf", "row_mask:0xf"),
  ):
    mod_inst = RdnaParser(mod_asm).parse().statements[0]
    assert isinstance(mod_inst, RdnaInstruction)
    m_op = mod_inst.operands[-1]
    assert isinstance(m_op, RdnaModifier)
    assert m_op.name == expected_mod


def test_rdna_cst_printing_transformer_fallbacks_and_analyzer() -> None:
  """Verify RDNA CST stringification, syntax errors, `RdnaTransformer` branches, and `RdnaAnalyzer`."""
  for bad_asm in ("~", "!", "MOV ,"):
    with pytest.raises(ValueError, match="Unexpected"):
      RdnaParser(bad_asm).parse()

  with pytest.raises(ValueError, match="Invalid RDNA opcode"):
    RdnaInstruction(opcode="bad op", operands=[])

  full_txt = str(
    RdnaParser(f"main:{_NL}  v_add_f32 v[0:1], s[0:1], v42 ; comment{_NL}.directive arg{_NL}  s_branch label").parse()
  )
  assert "v_add_f32" in full_txt
  assert "main" in full_txt

  assert isinstance(c_SGPR(0), RdnaNode)
  assert isinstance(c_VGPR(0), RdnaNode)
  assert str(RdnaImmediate(value=42, is_hex=False)) == "42"
  assert str(RdnaImmediate(value=42, is_hex=True)) == "0x2a"
  assert str(RdnaImmediate(value=-42, is_hex=True)) == "-0x2a"
  assert str(RdnaVGPR(index=0, count=1)) == "v0"
  assert str(RdnaVGPR(index=0, count=2)) == "v[0:1]"
  assert str(RdnaSGPR(index=0, count=1)) == "s0"
  assert str(RdnaSGPR(index=0, count=2)) == "s[0:1]"
  assert "offset:4" in str(RdnaMemory(base=RdnaVGPR(index=0, count=1), offset=4))
  assert str(RdnaLabelRef(name="lbl")) == "lbl"

  mod_triv = RdnaModifier(name="mod")
  mod_triv.trailing_trivia = [Trivia(" ")]
  assert str(mod_triv) == "mod "

  op1 = RdnaImmediate(value=1, is_hex=False)
  op2 = RdnaImmediate(value=2, is_hex=False)
  op2.leading_trivia = [Trivia(" ")]
  assert "MOV 1 2" in str(RdnaInstruction(opcode="MOV", operands=[op1, op2]))

  # `_get_trivia` and `RdnaTransformer` edge cases
  class _NodeWithTrivia:
    """Stub node carrying `leading_trivia`."""

    def __init__(self) -> None:
      """Set `leading_trivia`."""
      self.leading_trivia = [Trivia("mytrivia")]

  assert _get_trivia(_NodeWithTrivia()) == [Trivia("mytrivia")]

  t = RdnaTransformer()
  eof_tok = RdnaToken("EOF_TRIVIA", "")
  eof_tok.leading_trivia = []
  assert t.module([eof_tok]).statements == []
  assert t.directive([Token("DOT", "."), Token("IDENTIFIER", "req")]).params == []
  assert t.directive([Token("DOT", "."), Token("IDENTIFIER", "req"), None]).params == []
  assert t.directive([Token("DOT", "."), Token("IDENTIFIER", "req"), "not_a_list"]).params == ["not_a_list"]

  from ml_switcheroo.exceptions import SyntaxParseError

  with pytest.raises(SyntaxParseError, match="Invalid RDNA register operand"):
    t.register([Token("REG", "bad_reg")])
  with pytest.raises(SyntaxParseError, match="Expected RdnaModule root"):
    RdnaParser("v_nop")._validate_module(RdnaLabel(name="not_mod"))

  # `RdnaAnalyzer` loop limit metadata extraction
  lin_insts = [s for s in RdnaParser("s_cmp_lt_i32 s0, 5").parse().statements if isinstance(s, RdnaInstruction)]
  conv_insts = [s for s in RdnaParser("s_cmp_lt_i32 s0, 3").parse().statements if isinstance(s, RdnaInstruction)]
  add_insts = [s for s in RdnaParser("v_add_f32 v0, v1, v2").parse().statements if isinstance(s, RdnaInstruction)]
  assert RdnaAnalyzer.analyze_block("Linear", lin_insts).get("in_features") == 5
  assert RdnaAnalyzer.analyze_block("Conv2d", conv_insts).get("k") == 3
  assert "in_features" not in RdnaAnalyzer.analyze_block("Linear", add_insts)
