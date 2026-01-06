"""
Tests for SASS AST Node Representation.

Verifies:
1. Instruction formatting (Opcode, Operands, Semicolon).
2. Predicate formatting (Guard syntax, Negation).
3. Operand formatting (Registers, Immediates, Memory).
4. Structural nodes (Labels, Directives, Comments).
"""

from ml_switcheroo.core.sass.nodes import (
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
  """Verify standard, negated, and absolute register formats."""
  r0 = Register(name="R0")
  assert str(r0) == "R0"

  neg_r1 = Register(name="R1", negated=True)
  assert str(neg_r1) == "-R1"

  abs_r2 = Register(name="R2", absolute=True)
  assert str(abs_r2) == "|R2|"

  neg_abs_r3 = Register(name="R3", negated=True, absolute=True)
  assert str(neg_abs_r3) == "-|R3|"


def test_predicate_formatting():
  """Verify predicate syntax."""
  p0 = Predicate(name="P0")
  assert str(p0) == "P0"

  not_p1 = Predicate(name="P1", negated=True)
  assert str(not_p1) == "!P1"


def test_immediate_formatting():
  """Verify integer, float, and hex literals."""
  i1 = Immediate(value=1)
  assert str(i1) == "1"

  f1 = Immediate(value=1.5)
  assert str(f1) == "1.5"

  h1 = Immediate(value=255, is_hex=True)
  assert str(h1) == "0xff"


def test_memory_formatting():
  """Verify constant bank and global addressing."""
  # Constant Bank
  const_mem = Memory(base="c[0x0]", offset=0x4)
  assert str(const_mem) == "c[0x0][0x4]"

  const_mem_no_offset = Memory(base="c[0x1]")
  assert str(const_mem_no_offset) == "c[0x1][0x0]"

  # Global/Local with Register
  r1 = Register(name="R1")
  reg_mem = Memory(base=r1)
  assert str(reg_mem) == "[R1]"

  reg_mem_off = Memory(base=r1, offset=0x8)
  assert str(reg_mem_off) == "[R1 + 0x8]"


def test_instruction_formatting_basic():
  """Verify simple opcode + operands string."""
  inst = Instruction(opcode="FADD", operands=[Register("R0"), Register("R1"), Register("R2")])
  assert str(inst) == "FADD R0, R1, R2;"


def test_instruction_with_predicate():
  """Verify instruction with guard predicate."""
  pred = Predicate(name="P0", negated=True)
  inst = Instruction(opcode="MOV", operands=[Register("R0"), Register("RZ")], predicate=pred)
  # Note: Predicates on instructions are prefixed with @
  assert str(inst) == "@!P0 MOV R0, RZ;"


def test_label_formatting():
  """Verify label string generation."""
  lbl = Label(name="L_EXIT")
  assert str(lbl) == "L_EXIT:"


def test_directive_formatting():
  """Verify assembler directives."""
  d = Directive(name="headerflags", params=["@0x100"])
  assert str(d) == r".headerflags @0x100"

  d_no_args = Directive(name="foo")
  assert str(d_no_args) == ".foo"


def test_comment_formatting():
  """Verify comment syntax."""
  c = Comment(text="This is a loop")
  assert str(c) == "// This is a loop"
