"""Test suite for the Rdna Nodes module."""

from ml_switcheroo.core.compiler.frontends.rdna.cst import (
  RdnaComment,
  RdnaDirective,
  RdnaImmediate,
  RdnaInstruction,
  RdnaLabel,
  RdnaMemory,
  RdnaModifier,
  RdnaSGPR,
  RdnaVGPR,
)


def test_sgpr_formatting() -> None:
  """Verifies the behavior of sgpr formatting."""
  s0 = RdnaSGPR(index=0)
  assert str(s0) == "s0"
  s_range = RdnaSGPR(index=4, count=4)
  assert str(s_range) == "s[4:7]"


def test_vgpr_formatting() -> None:
  """Verifies the behavior of vgpr formatting."""
  v1 = RdnaVGPR(index=1)
  assert str(v1) == "v1"
  v_range = RdnaVGPR(index=0, count=3)
  assert str(v_range) == "v[0:2]"


def test_immediate_formatting() -> None:
  """Verifies the behavior of immediate formatting."""
  i1 = RdnaImmediate(value=42)
  assert str(i1) == "42"
  h1 = RdnaImmediate(value=255, is_hex=True)
  assert str(h1) == "0xff"
  f1 = RdnaImmediate(value=0.5)
  assert str(f1) == "0.5"


def test_modifier_formatting() -> None:
  """Verifies the behavior of modifier formatting."""
  mod = RdnaModifier(name="glc")
  assert str(mod) == "glc"


def test_memory_formatting() -> None:
  """Verifies the behavior of memory formatting."""
  base = RdnaVGPR(index=0)
  mem = RdnaMemory(base=base)
  assert str(mem) == "v0"
  mem_off = RdnaMemory(base=base, offset=16)
  assert str(mem_off) == "v0 offset:16"


def test_instruction_formatting() -> None:
  """Verifies the behavior of instruction formatting."""
  inst = RdnaInstruction(opcode="v_add_f32", operands=[RdnaVGPR(index=0), RdnaVGPR(index=1), RdnaVGPR(index=2)])
  assert str(inst) == "v_add_f32 v0, v1, v2"


def test_instruction_with_modifiers() -> None:
  """Verifies the behavior of instruction with modifiers."""
  inst = RdnaInstruction(
    opcode="global_load_dword",
    operands=[RdnaVGPR(index=1), RdnaVGPR(index=2), RdnaModifier(name="off"), RdnaModifier(name="glc")],
  )
  assert str(inst) == "global_load_dword v1, v2, off, glc"


def test_label_formatting() -> None:
  """Verifies the behavior of label formatting."""
  lbl = RdnaLabel(name="L_LOOP")
  assert str(lbl) == "L_LOOP:"


def test_directive_formatting() -> None:
  """Verifies the behavior of directive formatting."""
  d = RdnaDirective(name="text")
  assert str(d) == ".text"
  d_params = RdnaDirective(name="globl", params=["func_name"])
  assert str(d_params) == ".globl func_name"


def test_comment_formatting() -> None:
  """Verifies the behavior of comment formatting."""
  c = RdnaComment(text="Input: x")
  assert str(c) == "; Input: x"
