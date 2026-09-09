"""Test suite for the Sass Parser Coverage module."""

import typing

import pytest

from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassMemory, NvidiaSassModule, NvidiaSassRegister
from ml_switcheroo.core.compiler.frontends.nvidia_sass.parser import NvidiaSassParser


def test_nvidia_sass_parser_missing() -> None:
  """Verifies the behavior of NVIDIA_SASS parser missing."""
  parser = NvidiaSassParser(".text\n.global main")
  parser.parse()
  parser = NvidiaSassParser("MOV R0, R1\n.text")
  parser.parse()


def test_nvidia_sass_parser_error() -> None:
  """Docstring."""
  with pytest.raises(ValueError, match="Unexpected"):
    NvidiaSassParser("~").parse()

  with pytest.raises(ValueError):
    NvidiaSassParser("MOV ¿").parse()


def test_nvidia_sass_parser_empty() -> None:
  """Docstring."""
  mod: NvidiaSassModule = NvidiaSassParser("   ").parse()
  assert len(mod.statements) == 0


def test_nvidia_sass_parser_directive_no_params() -> None:
  """Docstring."""
  mod: NvidiaSassModule = NvidiaSassParser(".text").parse()
  assert mod.statements[0].name == "text"


def test_nvidia_sass_parser_directive_string() -> None:
  """Docstring."""
  mod: NvidiaSassModule = NvidiaSassParser('.headerflags @"NV_PROFILE"').parse()
  assert mod.statements[0].params[0] == '@"NV_PROFILE"'

  mod = NvidiaSassParser('.headerflags "NV_PROFILE"').parse()
  assert mod.statements[0].params[0] == '"NV_PROFILE"'


def test_nvidia_sass_parser_comments() -> None:
  """Docstring."""
  mod: NvidiaSassModule = NvidiaSassParser("// foo").parse()
  assert mod.statements[0].text == "foo"


def test_nvidia_sass_parser_empty_stmt() -> None:
  """Docstring."""
  mod: NvidiaSassModule = NvidiaSassParser(";").parse()
  assert len(mod.statements) == 0


def test_nvidia_sass_parser_mem_reg() -> None:
  """Docstring."""
  mod: NvidiaSassModule = NvidiaSassParser("LDG R0, [R1]").parse()
  assert getattr(mod.statements[0].operands[1], "offset", "fake") is None


def test_nvidia_sass_parser_predicate_operand() -> None:
  """Docstring."""
  mod: NvidiaSassModule = NvidiaSassParser("ISETP.NE.AND P0, PT, R1, 0x0, PT").parse()
  assert mod.statements[0].operands[-1].name == "PT"


def test_nvidia_sass_parser_at_string() -> None:
  """Docstring."""
  mod: NvidiaSassModule = NvidiaSassParser('.headerflags @"foo"').parse()
  assert mod.statements[0].params[0] == '@"foo"'


def test_nvidia_sass_parser_immediate() -> None:
  """Docstring."""
  mod: NvidiaSassModule = NvidiaSassParser("MOV R0, 5").parse()
  assert mod.statements[0].operands[1].value == 5
  assert mod.statements[0].operands[1].is_hex is False

  mod = NvidiaSassParser("MOV R0, 5.0").parse()
  assert mod.statements[0].operands[1].value == 5.0
  assert mod.statements[0].operands[1].is_hex is False


def test_nvidia_sass_parser_register() -> None:
  """Docstring."""
  mod: NvidiaSassModule = NvidiaSassParser("MOV R0, -R1").parse()
  assert mod.statements[0].operands[1].negated is True

  mod = NvidiaSassParser("MOV R0, |-R1|").parse()
  assert mod.statements[0].operands[1].absolute is True
  assert mod.statements[0].operands[1].negated is True


def test_nvidia_sass_parser_mem_bank() -> None:
  """Docstring."""
  mod: NvidiaSassModule = NvidiaSassParser("FADD R0, R1, c[0x0][0x4]").parse()
  assert mod.statements[0].operands[2].offset == 4


def test_nvidia_sass_parser_mem_offset() -> None:
  """Docstring."""
  mod: NvidiaSassModule = NvidiaSassParser("LDG.E R0, [R1 + 0x8]").parse()
  assert mod.statements[0].operands[1].offset == 8

  mod = NvidiaSassParser("LDG.E R0, [R1 + 8]").parse()
  assert mod.statements[0].operands[1].offset == 8


def test_nvidia_sass_parser_predicate_guard() -> None:
  """Docstring."""
  mod: NvidiaSassModule = NvidiaSassParser("@!P0 MOV R0, R1").parse()
  assert mod.statements[0].predicate.negated is True
  assert mod.statements[0].predicate.name == "P0"


def test_missing_coverage_sass() -> None:
  """Docstring."""
  # label
  mod: NvidiaSassModule = NvidiaSassParser("main:").parse()
  assert mod.statements[0].name == "main"

  # mem reg only
  mod = NvidiaSassParser("LDG R0, [R1]").parse()
  assert mod.statements[0].operands[1].offset is None

  # predicate operand with negation
  mod = NvidiaSassParser("ISETP.NE.AND P0, PT, R1, 0x0, !PT").parse()
  assert mod.statements[0].operands[-1].negated is True

  # operands single
  mod = NvidiaSassParser("RET").parse()
  assert len(mod.statements[0].operands) == 0

  # unreached token mismatch
  with pytest.raises(ValueError, match="Unexpected"):
    NvidiaSassParser("?").parse()


def test_missing_coverage_sass_2() -> None:
  """Docstring."""
  # directive list param
  mod: NvidiaSassModule = NvidiaSassParser('.headerflags @"NV_PROFILE", "OTHER"').parse()
  assert len(mod.statements[0].params) == 2

  mod = NvidiaSassParser('.headerflags @"NV_PROFILE", @"OTHER"').parse()
  assert mod.statements[0].params[1] == '@"OTHER"'


def test_missing_coverage_sass_3() -> None:
  """Docstring."""
  # directive param fallback list and Token fallback
  mod: NvidiaSassModule = NvidiaSassParser(".req 5").parse()  # number identifier?
  assert mod.statements[0].params[0] == "5"

  mod = NvidiaSassParser('.req foo, 5, @"test"').parse()
  assert len(mod.statements[0].params) == 3


def test_missing_coverage_sass_4() -> None:
  """Docstring."""
  # label trivia
  mod: NvidiaSassModule = NvidiaSassParser("main:\n  MOV R0, R1").parse()
  assert mod.statements[0].name == "main"

  # predicate missing coverage (no exclamation)
  mod = NvidiaSassParser("@P0 MOV R0, R1").parse()
  assert mod.statements[0].predicate.negated is False

  # mem reg fallback
  mod = NvidiaSassParser("LDG R0, [R1]").parse()
  assert mod.statements[0].operands[1].offset is None


def test_missing_coverage_sass_5() -> None:
  """Docstring."""
  # label operand
  mod: NvidiaSassModule = NvidiaSassParser("BRA main").parse()
  assert mod.statements[0].operands[0].name == "main"


def test_missing_coverage_sass_6() -> None:
  """Docstring."""
  # at_string parsing direct string value fallback
  mod: NvidiaSassModule = NvidiaSassParser('.headerflags "NV_PROFILE"').parse()
  assert mod.statements[0].params[0] == '"NV_PROFILE"'


def test_missing_coverage_sass_7() -> None:
  """Docstring."""
  mod: NvidiaSassModule = NvidiaSassParser(".headerflags 5").parse()
  assert mod.statements[0].params[0] == "5"

  mod = NvidiaSassParser("MOV R0").parse()
  assert len(mod.statements[0].operands) == 1

  mod = NvidiaSassParser("main:").parse()
  assert mod.statements[0].name == "main"


def test_nvidia_sass_parser_pred_at_bang_id() -> None:
  """Docstring."""
  mod: NvidiaSassModule = NvidiaSassParser("FADD R0, @!P0").parse()
  assert mod.statements[0].operands[1].negated is True
  assert mod.statements[0].operands[1].name == "P0"


def test_nvidia_sass_parser_pred_at_id() -> None:
  """Docstring."""
  mod: NvidiaSassModule = NvidiaSassParser("FADD R0, @P0").parse()
  assert mod.statements[0].operands[1].negated is False


def test_nvidia_sass_parser_pred_bang_reg() -> None:
  """Docstring."""
  mod: NvidiaSassModule = NvidiaSassParser("FADD R0, !PT").parse()
  assert mod.statements[0].operands[1].negated is True


def test_nvidia_sass_parser_pred_reg() -> None:
  """Docstring."""
  mod: NvidiaSassModule = NvidiaSassParser("FADD R0, PT").parse()
  assert isinstance(mod.statements[0].operands[1], NvidiaSassRegister)


def test_nvidia_sass_parser_mem_bank_2() -> None:
  """Docstring."""
  mod: NvidiaSassModule = NvidiaSassParser("FADD R0, c[0x0][0x4]").parse()
  assert isinstance(mod.statements[0].operands[1], NvidiaSassMemory)
  assert mod.statements[0].operands[1].offset == 4


def test_nvidia_sass_parser_mem_reg_neg_offset() -> None:
  """Docstring."""
  mod: NvidiaSassModule = NvidiaSassParser("FADD R0, [R1 - 0x4]").parse()
  assert mod.statements[0].operands[1].offset == -4


def test_nvidia_sass_parser_missing_coverage_8() -> None:
  """Docstring."""
  mod: NvidiaSassModule = NvidiaSassParser("FADD R0, @!R1").parse()
  assert mod.statements[0].operands[1].negated is True

  mod = NvidiaSassParser("FADD R0, @R1").parse()
  assert mod.statements[0].operands[1].negated is False

  mod = NvidiaSassParser("FADD R0, !P1").parse()
  assert mod.statements[0].operands[1].negated is True


def test_nvidia_sass_parser_missing_coverage_9() -> None:
  """Docstring."""
  # hit line 104 (node missing leading_trivia but has children)
  from ml_switcheroo.core.compiler.frontends.nvidia_sass.parser import _get_trivia

  class DummyChild:
    """A dummy child node."""

    def __init__(self) -> None:
      """Initializes DummyChild."""
      self.leading_trivia: list[typing.Any] = ["trivia"]

  class DummyNode:
    """A dummy node."""

    def __init__(self) -> None:
      """Initializes DummyNode."""
      self.children: list[typing.Any] = [DummyChild()]
      self.leading_trivia: list[typing.Any] = ["mytrivia"]

  assert _get_trivia(DummyNode()) == ["mytrivia"]  # type: ignore

  # test directive param list fallback (line 240, 242)
  # Not sure exactly how to hit it from parser, so we'll mock or force it via parser
  from lark import Token, Tree

  from ml_switcheroo.core.compiler.frontends.nvidia_sass.parser import NvidiaSassTransformer

  transformer = NvidiaSassTransformer()

  # line 242 (param_list is not list)
  d = transformer.directive([Token("DOT", "."), Token("IDENTIFIER", "req"), "string_param"])
  assert d.params == ["string_param"]

  # line 240 (param inside list is not Token or at_string or list)
  d = transformer.directive([Token("DOT", "."), Token("IDENTIFIER", "req"), [Token("IDENTIFIER", "abc"), 123]])
  assert d.params == ["abc", "123"]

  # line 234 (at_string tree)
  at_str = Tree("at_string", [Token("AT", "@"), Token("STRING", '"val"')])
  d = transformer.directive([Token("DOT", "."), Token("IDENTIFIER", "req"), [at_str]])
  assert d.params == ['@"val"']


def test_nvidia_sass_parser_branch_coverage() -> None:
  """Docstring."""
  from ml_switcheroo.core.compiler.frontends.nvidia_sass.parser import NvidiaSassTransformer

  transformer = NvidiaSassTransformer()
  # instruction with all None children (300->305)
  i = transformer.instruction([None, None])
  assert i.opcode == ""


def test_nvidia_sass_cst_print() -> None:
  """Docstring."""
  mod: NvidiaSassModule = NvidiaSassParser("main:\n  MOV R0, R1\n.text\n// foo\n").parse()
  assert "MOV" in str(mod)
  assert "main:" in str(mod)
  assert ".text" in str(mod)

  from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import (
    NvidiaSassImmediate,
    NvidiaSassMemory,
    NvidiaSassPredicate,
    NvidiaSassRegister,
  )

  imm = NvidiaSassImmediate(value=42, is_hex=False)
  assert str(imm) == "42"
  imm_hex = NvidiaSassImmediate(value=42, is_hex=True)
  assert str(imm_hex) == "0x2a"
  imm_neg = NvidiaSassImmediate(value=-42, is_hex=False)
  assert str(imm_neg) == "-42"
  imm_neg_hex = NvidiaSassImmediate(value=-42, is_hex=True)
  assert str(imm_neg_hex) == "-0x2a"

  pred = NvidiaSassPredicate(name="PT", negated=False)
  assert str(pred) == "PT"
  pred_neg = NvidiaSassPredicate(name="PT", negated=True)
  assert str(pred_neg) == "!PT"

  reg = NvidiaSassRegister(name="R0", negated=False, absolute=False)
  assert str(reg) == "R0"
  reg_neg = NvidiaSassRegister(name="R0", negated=True)
  assert str(reg_neg) == "-R0"
  reg_abs = NvidiaSassRegister(name="R0", absolute=True)
  assert str(reg_abs) == "|R0|"
  reg_abs_neg = NvidiaSassRegister(name="R0", absolute=True, negated=True)
  assert str(reg_abs_neg) == "-|R0|"

  mem = NvidiaSassMemory(base=reg, offset=0)
  assert "[R0]" in str(mem)
  mem2 = NvidiaSassMemory(base=reg, offset=4)
  assert "[R0 + 0x4]" in str(mem2)
  mem3 = NvidiaSassMemory(base=reg, offset=-4)
  assert "[R0 + -0x4]" in str(mem3)

  mem_bank = NvidiaSassMemory(base="c[0x0]", offset=8)
  assert "c[0x0][0x8]" in str(mem_bank)

  mem_bank_zero = NvidiaSassMemory(base="c[0x0]", offset=None)  # type: ignore
  assert "c[0x0][0x0]" in str(mem_bank_zero)


def test_nvidia_sass_parser_mem_bank_single() -> None:
  """Docstring."""
  mod: NvidiaSassModule = NvidiaSassParser("FADD R0, c[0x0]").parse()
  assert mod.statements[0].operands[1].offset is None


def test_nvidia_sass_parser_semi_instruction() -> None:
  """Docstring."""
  mod: NvidiaSassModule = NvidiaSassParser("MOV R0, R1;").parse()
  assert mod.statements[0].opcode == "MOV"


def test_nvidia_sass_cst_extra_coverage() -> None:
  """Docstring."""
  import pytest

  from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import (
    NvidiaSassDirective,
    NvidiaSassImmediate,
    NvidiaSassInstruction,
    NvidiaSassPredicate,
  )
  from ml_switcheroo.core.cst.base import Trivia

  with pytest.raises(ValueError, match="Invalid NVIDIA_SASS opcode"):
    NvidiaSassInstruction(opcode="BAD OPCODE", operands=[])

  pred = NvidiaSassPredicate(name="PT")
  pred.is_guard = True
  inst = NvidiaSassInstruction(opcode="MOV", operands=[], predicate=pred)
  assert "@PT MOV;" in str(inst)

  op1 = NvidiaSassImmediate(value=1, is_hex=False)
  op2 = NvidiaSassImmediate(value=2, is_hex=False)
  op2.leading_trivia = [Trivia(" ")]
  inst2 = NvidiaSassInstruction(opcode="MOV", operands=[op1, op2])
  assert "MOV 1 2;" in str(inst2)

  d = NvidiaSassDirective(name="text", params=["a", "b"])
  assert ".text a, b" in str(d)
