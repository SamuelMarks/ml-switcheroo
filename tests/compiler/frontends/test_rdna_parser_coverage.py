"""Docstring."""

import typing

import pytest

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
from ml_switcheroo.core.compiler.frontends.rdna.parser import RdnaParser


def test_rdna_parser_empty() -> None:
  """Docstring."""
  mod: RdnaModule = RdnaParser("  \n").parse()
  assert len(mod.statements) == 0


def test_rdna_parser_comment() -> None:
  """Docstring."""
  mod: RdnaModule = RdnaParser("; hello world").parse()
  assert len(mod.statements) == 1
  assert isinstance(mod.statements[0], RdnaComment)
  assert mod.statements[0].text == " hello world"


def test_rdna_parser_directive() -> None:
  """Docstring."""
  mod: RdnaModule = RdnaParser(".text").parse()
  assert isinstance(mod.statements[0], RdnaDirective)
  assert mod.statements[0].name == "text"
  assert len(mod.statements[0].params) == 0


def test_rdna_parser_directive_params() -> None:
  """Docstring."""
  mod: RdnaModule = RdnaParser('.global main, 5, 0x1, -5, -0x1, +5, +0x1, off, "str"').parse()
  assert isinstance(mod.statements[0], RdnaDirective)
  assert mod.statements[0].name == "global"
  assert len(mod.statements[0].params) == 9


def test_rdna_parser_label() -> None:
  """Docstring."""
  mod: RdnaModule = RdnaParser("main:").parse()
  assert isinstance(mod.statements[0], RdnaLabel)
  assert mod.statements[0].name == "main"


def test_rdna_parser_instruction_no_operands() -> None:
  """Docstring."""
  mod: RdnaModule = RdnaParser("s_waitcnt").parse()
  assert isinstance(mod.statements[0], RdnaInstruction)
  assert mod.statements[0].opcode == "s_waitcnt"
  assert len(mod.statements[0].operands) == 0


def test_rdna_parser_instruction_operands() -> None:
  """Docstring."""
  mod: RdnaModule = RdnaParser("v_add_f32 v0, v1, v2").parse()
  assert isinstance(mod.statements[0], RdnaInstruction)
  assert mod.statements[0].opcode == "v_add_f32"
  assert len(mod.statements[0].operands) == 3


def test_rdna_parser_memory() -> None:
  """Docstring."""
  mod: RdnaModule = RdnaParser("s_load_dword s0, [s[0:1]]").parse()
  assert isinstance(mod.statements[0], RdnaInstruction)
  assert isinstance(mod.statements[0].operands[1], RdnaMemory)
  assert getattr(mod.statements[0].operands[1], "offset", None) == 0

  mod = RdnaParser("s_load_dword s0, [s[0:1] + 0x4]").parse()
  assert getattr(mod.statements[0].operands[1], "offset", None) == 4

  mod = RdnaParser("s_load_dword s0, [s[0:1] - 0x4]").parse()
  assert getattr(mod.statements[0].operands[1], "offset", None) == -4


def test_rdna_parser_registers() -> None:
  """Docstring."""
  mod: RdnaModule = RdnaParser("v_add_f32 v[0:1], s[0:1], v42").parse()
  assert isinstance(mod.statements[0], RdnaInstruction)
  assert isinstance(mod.statements[0].operands[0], RdnaVGPR)
  assert mod.statements[0].operands[0].index == 0
  assert mod.statements[0].operands[0].count == 2

  assert isinstance(mod.statements[0].operands[1], RdnaSGPR)
  assert mod.statements[0].operands[1].index == 0
  assert mod.statements[0].operands[1].count == 2

  # Hit missing branch for singular register
  mod2: RdnaModule = RdnaParser("v_add_f32 v10, s20").parse()
  assert isinstance(mod2.statements[0], RdnaInstruction)
  assert isinstance(mod2.statements[0].operands[0], RdnaVGPR)
  assert mod2.statements[0].operands[0].index == 10
  assert mod2.statements[0].operands[0].count == 1
  assert isinstance(mod2.statements[0].operands[1], RdnaSGPR)
  assert mod2.statements[0].operands[1].index == 20
  assert mod2.statements[0].operands[1].count == 1


def test_rdna_parser_immediate() -> None:
  """Docstring."""
  mod: RdnaModule = RdnaParser("v_mov_b32 v0, 5").parse()
  assert isinstance(mod.statements[0], RdnaInstruction)
  imm = typing.cast(RdnaImmediate, mod.statements[0].operands[1])
  assert imm.value == 5
  assert not imm.is_hex

  mod = RdnaParser("v_mov_b32 v0, 5.0").parse()
  imm = typing.cast(RdnaImmediate, mod.statements[0].operands[1])  # type: ignore
  assert imm.value == 5.0

  mod = RdnaParser("v_mov_b32 v0, 0x5").parse()
  imm = typing.cast(RdnaImmediate, mod.statements[0].operands[1])  # type: ignore
  assert imm.value == 5
  assert imm.is_hex

  mod = RdnaParser("v_mov_b32 v0, -5").parse()
  imm = typing.cast(RdnaImmediate, mod.statements[0].operands[1])  # type: ignore
  assert imm.value == -5

  mod = RdnaParser("v_mov_b32 v0, -5.0").parse()
  imm = typing.cast(RdnaImmediate, mod.statements[0].operands[1])  # type: ignore
  assert imm.value == -5.0

  mod = RdnaParser("v_mov_b32 v0, -0x5").parse()
  imm = typing.cast(RdnaImmediate, mod.statements[0].operands[1])  # type: ignore
  assert imm.value == -5

  mod = RdnaParser("v_mov_b32 v0, +5").parse()
  imm = typing.cast(RdnaImmediate, mod.statements[0].operands[1])  # type: ignore
  assert imm.value == 5

  mod = RdnaParser("v_mov_b32 v0, +5.0").parse()
  imm = typing.cast(RdnaImmediate, mod.statements[0].operands[1])  # type: ignore
  assert imm.value == 5.0

  mod = RdnaParser("v_mov_b32 v0, +0x5").parse()
  imm = typing.cast(RdnaImmediate, mod.statements[0].operands[1])  # type: ignore
  assert imm.value == 5


def test_rdna_parser_modifier() -> None:
  """Docstring."""
  mod: RdnaModule = RdnaParser("s_waitcnt vmcnt(0)").parse()
  assert isinstance(mod.statements[0], RdnaInstruction)
  assert isinstance(mod.statements[0].operands[0], RdnaModifier)
  assert mod.statements[0].operands[0].name == "vmcnt(0)"

  mod = RdnaParser("v_add_f32 v0, v1, off").parse()
  assert isinstance(mod.statements[0], RdnaInstruction)
  assert isinstance(mod.statements[0].operands[2], RdnaModifier)
  assert mod.statements[0].operands[2].name == "off"


def test_rdna_parser_label_ref() -> None:
  """Docstring."""
  mod: RdnaModule = RdnaParser("s_branch main").parse()
  assert isinstance(mod.statements[0], RdnaInstruction)
  assert isinstance(mod.statements[0].operands[0], RdnaLabelRef)
  assert mod.statements[0].operands[0].name == "main"


def test_rdna_parser_error() -> None:
  """Docstring."""
  with pytest.raises(ValueError):
    RdnaParser("~").parse()


def test_rdna_nodes_cst_print() -> None:
  """Docstring."""
  # Hit str/repr lines in cst.py and nodes.py
  mod: RdnaModule = RdnaParser(
    "main:\n  v_add_f32 v[0:1], s[0:1], v42 ; comment\n.directive arg\n  s_branch label"
  ).parse()
  text: str = str(mod)
  assert "v_add_f32" in text
  assert "main" in text

  # Hit representations
  imm = RdnaImmediate(value=42, is_hex=False)
  assert str(imm) == "42"
  imm_hex = RdnaImmediate(value=42, is_hex=True)
  assert str(imm_hex) == "0x2a"
  imm_neg_hex = RdnaImmediate(value=-42, is_hex=True)
  assert str(imm_neg_hex) == "-0x2a"

  vgpr = RdnaVGPR(index=0, count=1)
  assert str(vgpr) == "v0"
  vgpr2 = RdnaVGPR(index=0, count=2)
  assert str(vgpr2) == "v[0:1]"

  sgpr = RdnaSGPR(index=0, count=1)
  assert str(sgpr) == "s0"
  sgpr2 = RdnaSGPR(index=0, count=2)
  assert str(sgpr2) == "s[0:1]"

  mem = RdnaMemory(base=sgpr, offset=0)
  assert str(mem) == "[s0]" or str(mem) == "s0" or "[s0]" in str(mem) or "s0" in str(mem)
  RdnaMemory(base=sgpr, offset=4)
  RdnaMemory(base=sgpr, offset=-4)

  lbl = RdnaLabelRef(name="lbl")
  assert str(lbl) == "lbl"


def test_cst_extra_coverage() -> None:
  """Docstring."""
  from ml_switcheroo.core.compiler.frontends.rdna.parser import _get_trivia

  class DummyChild:
    """Docstring."""

    def __init__(self) -> None:
      """Docstring."""
      self.leading_trivia: list[typing.Any] = ["trivia"]

  class DummyNode:
    """Docstring."""

    def __init__(self) -> None:
      """Docstring."""
      self.children: list[typing.Any] = [DummyChild()]
      self.leading_trivia: list[typing.Any] = ["mytrivia"]

  assert _get_trivia(DummyNode()) == ["mytrivia"]  # type: ignore


def test_rdna_extra_cst_coverage() -> None:
  """Docstring."""
  import pytest

  from ml_switcheroo.core.compiler.frontends.rdna.cst import (
    RdnaImmediate,
    RdnaInstruction,
    RdnaMemory,
    RdnaModifier,
    RdnaVGPR,
    c_SGPR,
    c_VGPR,
  )
  from ml_switcheroo.core.cst.base import Trivia

  assert isinstance(c_SGPR(0), RdnaNode)
  assert isinstance(c_VGPR(0), RdnaNode)

  mod = RdnaModifier(name="mod")
  mod.trailing_trivia = [Trivia(" ")]
  assert str(mod) == "mod "

  mem = RdnaMemory(base=RdnaVGPR(index=0, count=1), offset=4)
  assert "offset:4" in str(mem)

  with pytest.raises(ValueError, match="Invalid RDNA opcode"):
    RdnaInstruction(opcode="bad op", operands=[])

  op1 = RdnaImmediate(value=1, is_hex=False)
  op2 = RdnaImmediate(value=2, is_hex=False)
  op2.leading_trivia = [Trivia(" ")]
  inst = RdnaInstruction(opcode="MOV", operands=[op1, op2])
  assert "MOV 1 2" in str(inst)


def test_rdna_extra_analysis_coverage() -> None:
  """Docstring."""
  from ml_switcheroo.core.compiler.frontends.rdna.analysis import RdnaAnalyzer
  from ml_switcheroo.core.compiler.frontends.rdna.parser import RdnaParser

  # Linear mock
  # loop limits in analysis is taken from loops?
  # it says `if kind == "Linear": ...`
  metadata: dict[str, typing.Any] = RdnaAnalyzer.analyze_block(
    "Linear", RdnaParser("s_cmp_lt_i32 s0, 5").parse().statements
  )
  assert metadata.get("in_features") == 5
  metadata2: dict[str, typing.Any] = RdnaAnalyzer.analyze_block(
    "Conv2d", RdnaParser("s_cmp_lt_i32 s0, 3").parse().statements
  )
  assert metadata2.get("k") == 3


def test_rdna_parser_error_parse() -> None:
  """Docstring."""
  with pytest.raises(ValueError, match="Unexpected token"):
    RdnaParser("MOV ,").parse()


def test_rdna_analysis_empty() -> None:
  """Docstring."""
  from ml_switcheroo.core.compiler.frontends.rdna.analysis import RdnaAnalyzer
  from ml_switcheroo.core.compiler.frontends.rdna.parser import RdnaParser

  metadata: dict[str, typing.Any] = RdnaAnalyzer.analyze_block(
    "Linear", RdnaParser("v_add_f32 v0, v1, v2").parse().statements
  )
  assert "in_features" not in metadata
