"""Test suite for the Rdna Nodes module."""

from ml_switcheroo.core.compiler.frontends.rdna.nodes import (
  Comment,
  Directive,
  Immediate,
  Instruction,
  Label,
  Memory,
  Modifier,
  SGPR,
  VGPR,
)


def test_sgpr_formatting() -> None:
  """Verifies the behavior of sgpr formatting."""
  s0 = SGPR(index=0)
  assert str(s0) == "s0"
  s_range = SGPR(index=4, count=4)
  assert str(s_range) == "s[4:7]"


def test_vgpr_formatting() -> None:
  """Verifies the behavior of vgpr formatting."""
  v1 = VGPR(index=1)
  assert str(v1) == "v1"
  v_range = VGPR(index=0, count=3)
  assert str(v_range) == "v[0:2]"


def test_immediate_formatting() -> None:
  """Verifies the behavior of immediate formatting."""
  i1 = Immediate(value=42)
  assert str(i1) == "42"
  h1 = Immediate(value=255, is_hex=True)
  assert str(h1) == "0xff"
  f1 = Immediate(value=0.5)
  assert str(f1) == "0.5"


def test_modifier_formatting() -> None:
  """Verifies the behavior of modifier formatting."""
  mod = Modifier(name="glc")
  assert str(mod) == "glc"


def test_memory_formatting() -> None:
  """Verifies the behavior of memory formatting."""
  base = VGPR(0)
  mem = Memory(base=base)
  assert str(mem) == "v0"
  mem_off = Memory(base=base, offset=16)
  assert str(mem_off) == "v0 offset:16"


def test_instruction_formatting() -> None:
  """Verifies the behavior of instruction formatting."""
  inst = Instruction(opcode="v_add_f32", operands=[VGPR(0), VGPR(1), VGPR(2)])
  assert str(inst) == "v_add_f32 v0, v1, v2"


def test_instruction_with_modifiers() -> None:
  """Verifies the behavior of instruction with modifiers."""
  inst = Instruction(opcode="global_load_dword", operands=[VGPR(1), VGPR(2), Modifier("off"), Modifier("glc")])
  assert str(inst) == "global_load_dword v1, v2, off, glc"


def test_label_formatting() -> None:
  """Verifies the behavior of label formatting."""
  lbl = Label(name="L_LOOP")
  assert str(lbl) == "L_LOOP:"


def test_directive_formatting() -> None:
  """Verifies the behavior of directive formatting."""
  d = Directive(name="text")
  assert str(d) == ".text"
  d_params = Directive(name="globl", params=["func_name"])
  assert str(d_params) == ".globl func_name"


def test_comment_formatting() -> None:
  """Verifies the behavior of comment formatting."""
  c = Comment(text="Input: x")
  assert str(c) == "; Input: x"
