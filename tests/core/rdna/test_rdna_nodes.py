from ml_switcheroo.compiler.frontends.rdna.nodes import (
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
  """Verify scalar register and range syntax."""
  s0 = SGPR(index=0)
  assert str(s0) == "s0"

  s_range = SGPR(index=4, count=4)
  # s[4:7] means s4, s5, s6, s7 (4 regs starting at 4)
  assert str(s_range) == "s[4:7]"


def test_vgpr_formatting() -> None:
  """Verify vector register and range syntax."""
  v1 = VGPR(index=1)
  assert str(v1) == "v1"

  v_range = VGPR(index=0, count=3)
  assert str(v_range) == "v[0:2]"


def test_immediate_formatting() -> None:
  """Verify integer and hex literals."""
  i1 = Immediate(value=42)
  assert str(i1) == "42"

  h1 = Immediate(value=255, is_hex=True)
  assert str(h1) == "0xff"

  f1 = Immediate(value=0.5)
  assert str(f1) == "0.5"


def test_modifier_formatting() -> None:
  """Verify modifiers are strings."""
  mod = Modifier(name="glc")
  assert str(mod) == "glc"


def test_memory_formatting() -> None:
  """Verify logical memory operand syntax."""
  base = VGPR(0)
  # Plain register base
  mem = Memory(base=base)
  assert str(mem) == "v0"

  # With offset
  mem_off = Memory(base=base, offset=16)
  assert str(mem_off) == "v0 offset:16"


def test_instruction_formatting() -> None:
  """Verify instruction construction."""
  # v_add_f32 v0, v1, v2
  inst = Instruction(
    opcode="v_add_f32",
    operands=[VGPR(0), VGPR(1), VGPR(2)],
  )
  assert str(inst) == "v_add_f32 v0, v1, v2"


def test_instruction_with_modifiers() -> None:
  """Verify instruction with modifiers."""
  # global_load_dword v1, v2, off glc
  inst = Instruction(
    opcode="global_load_dword",
    operands=[VGPR(1), VGPR(2), Modifier("off"), Modifier("glc")],
  )
  assert str(inst) == "global_load_dword v1, v2, off, glc"


def test_label_formatting() -> None:
  """Verify label syntax."""
  lbl = Label(name="L_LOOP")
  assert str(lbl) == "L_LOOP:"


def test_directive_formatting() -> None:
  """Verify directive syntax."""
  d = Directive(name="text")
  assert str(d) == ".text"

  d_params = Directive(name="globl", params=["func_name"])
  assert str(d_params) == ".globl func_name"


def test_comment_formatting() -> None:
  """Verify RDNA comment style (;)."""
  c = Comment(text="Input: x")
  assert str(c) == "; Input: x"
