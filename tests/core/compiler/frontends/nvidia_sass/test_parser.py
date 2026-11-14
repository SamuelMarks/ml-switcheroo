"""Unit tests for `NvidiaSassParser`, `NvidiaSassTransformer`, and SASS CST (`ml_switcheroo.core.compiler.frontends.nvidia_sass`)."""

import pytest
from lark import Token, Tree

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
from ml_switcheroo.core.compiler.frontends.nvidia_sass.parser import (
  NvidiaSassParser,
  NvidiaSassTransformer,
  _get_trivia,
)
from ml_switcheroo.core.cst.base import Trivia

_NL = chr(10)


def test_nvidia_sass_parser_directives_predicates_operands_and_memory() -> None:
  """Verify `NvidiaSassParser` parses comments, labels, directives, predicates, registers, memory, and immediates."""
  assert NvidiaSassParser("").parse().statements == []
  assert NvidiaSassParser("   ").parse().statements == []
  assert NvidiaSassParser(";").parse().statements == []

  cmt = NvidiaSassParser("// comment").parse().statements[0]
  assert isinstance(cmt, NvidiaSassComment)
  assert cmt.text == "comment"

  lbl = NvidiaSassParser("L_1:").parse().statements[0]
  assert isinstance(lbl, NvidiaSassLabel)
  assert lbl.name == "L_1"

  d_bare = NvidiaSassParser(".text").parse().statements[0]
  assert isinstance(d_bare, NvidiaSassDirective)
  assert (d_bare.name, d_bare.params) == ("text", [])

  d_semi = NvidiaSassParser(".headerflags 1, 2;").parse().statements[0]
  assert isinstance(d_semi, NvidiaSassDirective)
  assert (d_semi.name, d_semi.params) == ("headerflags", ["1", "2"])

  for dir_src, expected_params in (
    ('.headerflags @"NV_PROFILE", "OTHER"', ['@"NV_PROFILE"', '"OTHER"']),
    ('.headerflags @"NV_PROFILE", @"OTHER"', ['@"NV_PROFILE"', '@"OTHER"']),
    ('.req foo, 5, @"test"', ["foo", "5", '@"test"']),
  ):
    d_item = NvidiaSassParser(dir_src).parse().statements[0]
    assert isinstance(d_item, NvidiaSassDirective)
    assert d_item.params == expected_params

  assert len(NvidiaSassParser(f".text{_NL}.global main").parse().statements) == 2
  assert len(NvidiaSassParser(f"MOV R0, R1{_NL}.text").parse().statements) == 2
  assert len(NvidiaSassParser(f"FADD;{_NL}NOP").parse().statements) == 2
  assert len(NvidiaSassParser(f"main:{_NL}  MOV R0, R1").parse().statements) == 2

  # Guarded instructions (`@P0`, `@!P1`) and zero-operand instructions (`NOP;`, `RET`)
  g_pos = NvidiaSassParser("@P0 FADD R0, R1;").parse().statements[0]
  assert isinstance(g_pos, NvidiaSassInstruction)
  assert g_pos.predicate is not None
  assert (g_pos.predicate.name, g_pos.predicate.negated) == ("P0", False)

  g_neg = NvidiaSassParser("@!P1 FADD R0, R1;").parse().statements[0]
  assert isinstance(g_neg, NvidiaSassInstruction)
  assert g_neg.predicate is not None
  assert (g_neg.predicate.name, g_neg.predicate.negated) == ("P1", True)

  ret_inst = NvidiaSassParser("RET").parse().statements[0]
  assert isinstance(ret_inst, NvidiaSassInstruction)
  assert ret_inst.operands == []

  # Rich 12-operand instruction + predicate/register/memory/immediate permutations
  rich = (
    NvidiaSassParser("FADD R0, -R1, |R2|, -|R3|, c[0x0][0x4], [R1 + 0x4], [R2], 0x1, 1.5, @P0, L_1, 5")
    .parse()
    .statements[0]
  )
  assert isinstance(rich, NvidiaSassInstruction)
  assert len(rich.operands) == 12
  r0, r1, r2, r3 = rich.operands[0], rich.operands[1], rich.operands[2], rich.operands[3]
  assert isinstance(r0, NvidiaSassRegister)
  assert r0.name == "R0"
  assert isinstance(r1, NvidiaSassRegister)
  assert r1.negated is True
  assert isinstance(r2, NvidiaSassRegister)
  assert r2.absolute is True
  assert isinstance(r3, NvidiaSassRegister)
  assert (r3.absolute, r3.negated) == (True, True)

  m_bank, m_pos, m_zero = rich.operands[4], rich.operands[5], rich.operands[6]
  assert isinstance(m_bank, NvidiaSassMemory)
  assert (m_bank.base, m_bank.offset) == ("c[0x0]", 4)
  assert isinstance(m_pos, NvidiaSassMemory)
  assert m_pos.offset == 4
  assert isinstance(m_zero, NvidiaSassMemory)
  assert m_zero.offset is None

  imm_hex, imm_flt, pred_op, lbl_op, imm_int = (
    rich.operands[7],
    rich.operands[8],
    rich.operands[9],
    rich.operands[10],
    rich.operands[11],
  )
  assert isinstance(imm_hex, NvidiaSassImmediate)
  assert (imm_hex.value, imm_hex.is_hex) == (1, True)
  assert isinstance(imm_flt, NvidiaSassImmediate)
  assert imm_flt.value == 1.5
  assert isinstance(pred_op, NvidiaSassPredicate)
  assert pred_op.name == "P0"
  assert isinstance(lbl_op, NvidiaSassLabel)
  assert lbl_op.name == "L_1"
  assert isinstance(imm_int, NvidiaSassImmediate)
  assert (imm_int.value, imm_int.is_hex) == (5, False)

  # Additional memory/predicate operand forms (`c[0x0]`, `[R1 + 10]`, `[R1 - 0x4]`, `@!P0`, `!PT`, `PT`, `@!R1`, `@R1`, `!P1`)
  for extra_asm in (
    "FADD R0, c[0x0]",
    "FADD [R1 + 10]",
    "FADD R0, [R1 - 0x4]",
    "FADD R0, @!P0",
    "FADD R0, !PT",
    "FADD R0, PT",
    "FADD R0, @!R1",
    "FADD R0, @R1",
    "FADD R0, !P1",
  ):
    parsed_extra = NvidiaSassParser(extra_asm).parse().statements[0]
    assert isinstance(parsed_extra, NvidiaSassInstruction)


def test_nvidia_sass_cst_printing_errors_and_transformer_helpers() -> None:
  """Verify SASS CST stringification, invalid opcodes/syntax, `_get_trivia`, and `NvidiaSassTransformer`."""
  for bad_src in ("~", "?", "MOV ¿"):
    with pytest.raises(ValueError, match="Unexpected"):
      NvidiaSassParser(bad_src).parse()

  with pytest.raises(ValueError, match="Invalid NVIDIA_SASS opcode"):
    NvidiaSassInstruction(opcode="BAD OPCODE", operands=[])

  mod_txt = str(NvidiaSassParser(f"main:{_NL}  MOV R0, R1{_NL}.text{_NL}// foo{_NL}").parse())
  assert "MOV" in mod_txt
  assert "main:" in mod_txt
  assert ".text" in mod_txt

  assert str(NvidiaSassImmediate(value=42, is_hex=False)) == "42"
  assert str(NvidiaSassImmediate(value=42, is_hex=True)) == "0x2a"
  assert str(NvidiaSassImmediate(value=-42, is_hex=False)) == "-42"
  assert str(NvidiaSassImmediate(value=-42, is_hex=True)) == "-0x2a"
  assert str(NvidiaSassPredicate(name="PT", negated=False)) == "PT"
  assert str(NvidiaSassPredicate(name="PT", negated=True)) == "!PT"

  reg = NvidiaSassRegister(name="R0")
  assert str(reg) == "R0"
  assert str(NvidiaSassRegister(name="R0", negated=True)) == "-R0"
  assert str(NvidiaSassRegister(name="R0", absolute=True)) == "|R0|"
  assert str(NvidiaSassRegister(name="R0", absolute=True, negated=True)) == "-|R0|"

  assert "[R0]" in str(NvidiaSassMemory(base=reg, offset=0))
  assert "[R0 + 0x4]" in str(NvidiaSassMemory(base=reg, offset=4))
  assert "[R0 + -0x4]" in str(NvidiaSassMemory(base=reg, offset=-4))
  assert "c[0x0][0x8]" in str(NvidiaSassMemory(base="c[0x0]", offset=8))
  assert "c[0x0][0x0]" in str(NvidiaSassMemory(base="c[0x0]", offset=None))

  guard_pred = NvidiaSassPredicate(name="PT", is_guard=True)
  assert "@PT MOV;" in str(NvidiaSassInstruction(opcode="MOV", operands=[], predicate=guard_pred))
  op1 = NvidiaSassImmediate(value=1, is_hex=False)
  op2 = NvidiaSassImmediate(value=2, is_hex=False)
  op2.leading_trivia = [Trivia(" ")]
  assert "MOV 1 2;" in str(NvidiaSassInstruction(opcode="MOV", operands=[op1, op2]))
  assert ".text a, b" in str(NvidiaSassDirective(name="text", params=["a", "b"]))

  # `_get_trivia` across direct trivia, empty children, missing children, and nested child trivia
  class _EmptyChildren:
    """Stub node with empty `children`."""

    def __init__(self) -> None:
      """Set empty `children`."""
      self.children: list[Token] = []

  class _NoChildren:
    """Stub node without `children` or `leading_trivia`."""

  class _ChildWithTrivia:
    """Stub node with `leading_trivia`."""

    def __init__(self) -> None:
      """Set `leading_trivia`."""
      self.leading_trivia = [Trivia("some_trivia")]

  assert _get_trivia(_EmptyChildren()) == []
  assert _get_trivia(_NoChildren()) == []
  assert _get_trivia(_ChildWithTrivia()) == [Trivia("some_trivia")]
  assert _get_trivia(Tree("dummy", [_ChildWithTrivia()])) == [Trivia("some_trivia")]

  # `NvidiaSassTransformer` non-list / mixed / `at_string` directive params and empty instruction
  t = NvidiaSassTransformer()
  assert t.directive([Token("DOT", "."), Token("IDENTIFIER", "req"), "string_param"]).params == ["string_param"]
  assert t.directive([Token("DOT", "."), Token("IDENTIFIER", "req"), [Token("IDENTIFIER", "abc"), 123]]).params == [
    "abc",
    "123",
  ]
  at_tree = Tree("at_string", [Token("AT", "@"), Token("STRING", '"val"')])
  assert t.directive([Token("DOT", "."), Token("IDENTIFIER", "req"), [at_tree]]).params == ['@"val"']
  assert t.instruction([None, None]).opcode == ""

  from ml_switcheroo.exceptions import SyntaxParseError

  with pytest.raises(SyntaxParseError, match="Expected NvidiaSassModule root"):
    NvidiaSassParser("NOP;")._validate_module(NvidiaSassLabel(name="not_mod"))
