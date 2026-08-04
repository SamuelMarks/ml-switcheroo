"""Tests for the RDNA Concrete Syntax Tree nodes."""

from ml_switcheroo.core.compiler.frontends.rdna.cst import (
  RdnaModule,
  RdnaInstruction,
  RdnaVGPR,
  RdnaSGPR,
  RdnaImmediate,
  RdnaMemory,
  RdnaLabel,
  RdnaLabelRef,
  RdnaDirective,
  RdnaComment,
  RdnaModifier,
  c_SGPR,
  c_VGPR,
)
from ml_switcheroo.core.cst.base import Trivia


def test_vgpr_to_text() -> None:
  """Test VGPR serialization."""
  reg1 = RdnaVGPR(index=0, leading_trivia=[Trivia(" ")])
  assert reg1.to_text() == " v0"
  reg2 = RdnaVGPR(index=1, count=3)
  assert reg2.to_text() == "v[1:3]"


def test_sgpr_to_text() -> None:
  """Test SGPR serialization."""
  reg1 = RdnaSGPR(index=4, trailing_trivia=[Trivia(",")])
  assert reg1.to_text() == "s4,"
  reg2 = RdnaSGPR(index=0, count=2)
  assert reg2.to_text() == "s[0:1]"


def test_helpers() -> None:
  """Test c_SGPR and c_VGPR helpers."""
  assert c_SGPR(5).to_text() == "s5"
  assert c_VGPR(10).to_text() == "v10"


def test_immediate_to_text() -> None:
  """Test immediate serialization."""
  imm1 = RdnaImmediate(value=42)
  assert imm1.to_text() == "42"
  imm2 = RdnaImmediate(value=255, is_hex=True)
  assert imm2.to_text() == "0xff"


def test_modifier_to_text() -> None:
  """Test modifier serialization."""
  mod = RdnaModifier(name="glc")
  assert mod.to_text() == "glc"


def test_memory_to_text() -> None:
  """Test memory operand serialization."""
  mem1 = RdnaMemory(base=RdnaVGPR(index=1))
  assert mem1.to_text() == "v1"
  mem2 = RdnaMemory(base=RdnaSGPR(index=4, count=2), offset=16)
  assert mem2.to_text() == "s[4:5] offset:16"


def test_label_ref_to_text() -> None:
  """Test label reference serialization."""
  lbl = RdnaLabelRef(name="target")
  assert lbl.to_text() == "target"


def test_instruction_to_text() -> None:
  """Test instruction serialization."""
  inst = RdnaInstruction(
    leading_trivia=[Trivia("  ")],
    opcode="v_add_f32",
    operands=[
      RdnaVGPR(index=0),
      RdnaVGPR(index=1, leading_trivia=[Trivia(" ")]),
      RdnaVGPR(index=2, leading_trivia=[Trivia(", ")]),
    ],
  )
  assert inst.to_text() == "  v_add_f32 v0 v1, v2"


def test_label_to_text() -> None:
  """Test label serialization."""
  lbl = RdnaLabel(name="L_START", trailing_trivia=[Trivia("\n")])
  assert lbl.to_text() == "L_START:\n"


def test_directive_to_text() -> None:
  """Test directive serialization."""
  dir1 = RdnaDirective(name="text")
  assert dir1.to_text() == ".text"
  dir2 = RdnaDirective(name="globl", params=["main"])
  assert dir2.to_text() == ".globl main"


def test_comment_to_text() -> None:
  """Test comment serialization."""
  com = RdnaComment(text="a comment")
  assert com.to_text() == "; a comment"


def test_module_to_text() -> None:
  """Test module serialization."""
  mod = RdnaModule(
    leading_trivia=[Trivia("\n")],
    statements=[
      RdnaLabel(name="main", trailing_trivia=[Trivia("\n")]),
      RdnaInstruction(leading_trivia=[Trivia("  ")], opcode="s_endpgm", trailing_trivia=[Trivia("\n")]),
    ],
  )
  assert mod.to_text() == "\nmain:\n  s_endpgm\n"
