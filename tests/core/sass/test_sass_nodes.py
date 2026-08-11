"""Test suite for the Sass Nodes module."""

from ml_switcheroo.core.compiler.frontends.sass.nodes import (
  Comment,
  Directive,
  Immediate,
  Instruction,
  Label,
  Memory,
  Predicate,
  Register,
)


def test_register_formatting():
  """Verifies the behavior of register formatting."""
  r0 = Register(name="R0")
  assert str(r0) == "R0"
  neg_r1 = Register(name="R1", negated=True)
  assert str(neg_r1) == "-R1"
  abs_r2 = Register(name="R2", absolute=True)
  assert str(abs_r2) == "|R2|"
  neg_abs_r3 = Register(name="R3", negated=True, absolute=True)
  assert str(neg_abs_r3) == "-|R3|"


def test_predicate_formatting():
  """Verifies the behavior of predicate formatting."""
  p0 = Predicate(name="P0")
  assert str(p0) == "P0"
  not_p1 = Predicate(name="P1", negated=True)
  assert str(not_p1) == "!P1"


def test_immediate_formatting():
  """Verifies the behavior of immediate formatting."""
  i1 = Immediate(value=1)
  assert str(i1) == "1"
  f1 = Immediate(value=1.5)
  assert str(f1) == "1.5"
  h1 = Immediate(value=255, is_hex=True)
  assert str(h1) == "0xff"


def test_memory_formatting():
  """Verifies the behavior of memory formatting."""
  const_mem = Memory(base="c[0x0]", offset=4)
  assert str(const_mem) == "c[0x0][0x4]"
  r1 = Register(name="R1")
  reg_mem = Memory(base=r1)
  assert str(reg_mem) == "[R1]"
  reg_mem_off = Memory(base=r1, offset=8)
  assert str(reg_mem_off) == "[R1 + 0x8]"


def test_instruction_formatting_basic():
  """Verifies the behavior of instruction formatting basic."""
  inst = Instruction(opcode="FADD", operands=[Register(name="R0"), Register(name="R1"), Register(name="R2")])
  assert str(inst) == "FADD R0, R1, R2;"


def test_instruction_with_predicate():
  """Verifies the behavior of instruction with predicate."""
  pred = Predicate(name="P0", negated=True)
  inst = Instruction(opcode="MOV", operands=[Register(name="R0"), Register(name="RZ")], predicate=pred)
  assert str(inst) == "@!P0 MOV R0, RZ;"


def test_label_formatting():
  """Verifies the behavior of label formatting."""
  lbl = Label(name="L_EXIT")
  assert str(lbl) == "L_EXIT:"


def test_directive_formatting():
  """Verifies the behavior of directive formatting."""
  d = Directive(name="headerflags", params=["@0x100"])
  assert str(d) == ".headerflags @0x100"


def test_comment_formatting():
  """Verifies the behavior of comment formatting."""
  c = Comment(text="This is a loop")
  assert str(c) == "// This is a loop"


def test_sass_nodes_immediate_float_hex():
  """Test sass nodes immediate float hex."""
  from ml_switcheroo.core.compiler.frontends.sass.nodes import Immediate

  imm = Immediate(value=1.5, is_hex=True)
  assert str(imm) == "0x1"


def test_sass_nodes_memory_no_offset():
  """Test sass nodes memory no offset."""
  from ml_switcheroo.core.compiler.frontends.sass.nodes import Memory

  mem = Memory(base="c[0x0]")
  assert str(mem) == "c[0x0][0x0]"


def test_sass_nodes_instruction_invalid_opcode():
  """Test sass nodes instruction invalid opcode."""
  from ml_switcheroo.core.compiler.frontends.sass.nodes import Instruction
  import pytest

  with pytest.raises(ValueError):
    Instruction(opcode="OP CODE")


def test_sass_nodes_instruction_no_operands():
  """Test sass nodes instruction no operands."""
  from ml_switcheroo.core.compiler.frontends.sass.nodes import Instruction

  inst = Instruction(opcode="RET")
  # 187-189
  assert str(inst) == "RET ;"


def test_sass_nodes_instruction_trailing_trivia():
  """Test sass nodes instruction trailing trivia."""
  from ml_switcheroo.core.compiler.frontends.sass.nodes import Instruction

  inst = Instruction(opcode="RET")
  inst.trailing_trivia = " "
  assert str(inst) == "RET  "


def test_sass_nodes_directive_no_params():
  """Test sass nodes directive no params."""
  from ml_switcheroo.core.compiler.frontends.sass.nodes import Directive

  d = Directive(name="text")
  assert str(d) == ".text"


def test_sass_nodes_base_node_str():
  """Test sass nodes base node str."""
  from ml_switcheroo.core.compiler.frontends.sass.nodes import SassNode

  class Dummy(SassNode):
    """Dummy."""

    def __str__(self):
      return super().__str__()

  assert super(Dummy, Dummy()).__str__() is None


def test_sass_nodes_instruction_operand_with_leading_trivia():
  """Test sass nodes instruction operand with leading trivia."""
  from ml_switcheroo.core.compiler.frontends.sass.nodes import Instruction, Register

  op1 = Register(name="R0")
  op1.leading_trivia = " "
  inst = Instruction(opcode="OP", operands=[op1])
  assert str(inst) == "OP R0;"
