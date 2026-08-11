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


def test_rdna_nodes_operand_base():
  """Test rdna nodes operand base."""
  from ml_switcheroo.core.compiler.frontends.rdna.nodes import Operand

  op = Operand()
  assert str(op) == ""


def test_rdna_nodes_instruction_with_leading_trivia():
  """Test rdna nodes instruction with leading trivia."""
  from ml_switcheroo.core.compiler.frontends.rdna.nodes import Instruction, SGPR

  op1 = SGPR(0)
  op1.leading_trivia = " "
  inst = Instruction("v_add_f32", [op1])
  assert str(inst) == "v_add_f32 s0"


def test_rdna_nodes_instruction_invalid_opcode():
  """Test rdna nodes instruction invalid opcode."""
  from ml_switcheroo.core.compiler.frontends.rdna.nodes import Instruction
  import pytest

  with pytest.raises(ValueError):
    Instruction("v_add f32", [])


def test_rdna_nodes_memory_zero_offset():
  """Test rdna nodes memory zero offset."""
  from ml_switcheroo.core.compiler.frontends.rdna.nodes import Memory, SGPR

  mem = Memory(SGPR(0), 0)
  assert str(mem) == "s0"


def test_rdna_nodes_directive_params():
  """Test rdna nodes directive params."""
  from ml_switcheroo.core.compiler.frontends.rdna.nodes import Directive

  d = Directive("text", ["a", "b"])
  assert str(d) == ".text a, b"


def test_rdna_nodes_immediate_hex():
  """Test rdna nodes immediate hex."""
  from ml_switcheroo.core.compiler.frontends.rdna.nodes import Immediate

  imm = Immediate(16, is_hex=True)
  assert str(imm) == "0x10"


def test_rdna_nodes_modifier():
  """Test rdna nodes modifier."""
  from ml_switcheroo.core.compiler.frontends.rdna.nodes import Modifier

  mod = Modifier("glc")
  assert str(mod) == "glc"


def test_rdna_nodes_memory_with_offset():
  """Test rdna nodes memory with offset."""
  from ml_switcheroo.core.compiler.frontends.rdna.nodes import Memory, SGPR

  mem = Memory(SGPR(0), 4)
  assert str(mem) == "s0 offset:4"


def test_rdna_nodes_label():
  """Test rdna nodes label."""
  from ml_switcheroo.core.compiler.frontends.rdna.nodes import Label

  lbl = Label("loop")
  assert str(lbl) == "loop:"


def test_rdna_nodes_directive_no_params():
  """Test rdna nodes directive no params."""
  from ml_switcheroo.core.compiler.frontends.rdna.nodes import Directive

  d = Directive("text")
  assert str(d) == ".text"


def test_rdna_nodes_comment():
  """Test rdna nodes comment."""
  from ml_switcheroo.core.compiler.frontends.rdna.nodes import Comment

  c = Comment("hi")
  assert str(c) == "; hi"


def test_rdna_nodes_instruction_no_operands():
  """Test rdna nodes instruction no operands."""
  from ml_switcheroo.core.compiler.frontends.rdna.nodes import Instruction

  inst = Instruction("s_endpgm")
  assert str(inst) == "s_endpgm"


def test_rdna_nodes_instruction_second_operand_leading_trivia():
  """Test rdna nodes instruction second operand leading trivia."""
  from ml_switcheroo.core.compiler.frontends.rdna.nodes import Instruction, SGPR

  op1 = SGPR(0)
  op2 = SGPR(1)
  op2.leading_trivia = " "
  inst = Instruction("v_add", [op1, op2])
  assert str(inst) == "v_add s0 s1"


def test_rdna_nodes_rdnanode_base():
  """Test rdna nodes rdnanode base."""
  from ml_switcheroo.core.compiler.frontends.rdna.nodes import RdnaNode

  class Dummy(RdnaNode):
    """Dummy."""

    def __str__(self):
      return super().__str__()

  assert str(Dummy()) == ""


def test_rdna_nodes_label_ref():
  """Test rdna nodes label ref."""
  from ml_switcheroo.core.compiler.frontends.rdna.nodes import LabelRef

  ref = LabelRef("L1")
  assert str(ref) == "L1"


def test_rdna_nodes_sgpr_count():
  """Test rdna nodes sgpr count."""
  from ml_switcheroo.core.compiler.frontends.rdna.nodes import SGPR

  s = SGPR(0, 4)
  assert str(s) == "s[0:3]"


def test_rdna_nodes_vgpr_count():
  """Test rdna nodes vgpr count."""
  from ml_switcheroo.core.compiler.frontends.rdna.nodes import VGPR

  v = VGPR(0, 4)
  assert str(v) == "v[0:3]"


def test_rdna_nodes_c_helpers():
  """Test rdna nodes c helpers."""
  from ml_switcheroo.core.compiler.frontends.rdna.nodes import c_SGPR, c_VGPR

  assert str(c_SGPR(1)) == "s1"
  assert str(c_VGPR(1)) == "v1"


def test_rdna_nodes_immediate_not_hex():
  """Test rdna nodes immediate not hex."""
  from ml_switcheroo.core.compiler.frontends.rdna.nodes import Immediate

  imm = Immediate(16, is_hex=False)
  assert str(imm) == "16"


def test_rdna_nodes_instruction_comma():
  """Test rdna nodes instruction comma."""
  from ml_switcheroo.core.compiler.frontends.rdna.nodes import Instruction, SGPR

  op1 = SGPR(0)
  op2 = SGPR(1)
  # op2.leading_trivia is empty by default
  inst = Instruction("v_add", [op1, op2])
  assert str(inst) == "v_add s0, s1"
