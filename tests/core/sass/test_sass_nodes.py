"""Test suite for the Sass Nodes module."""

from ml_switcheroo.core.compiler.frontends.sass.cst import (
  SassComment,
  SassDirective,
  SassImmediate,
  SassInstruction,
  SassLabel,
  SassMemory,
  SassPredicate,
  SassRegister,
)


def test_register_formatting():
  """Verifies the behavior of register formatting."""
  r0 = SassRegister(name="R0")
  assert str(r0) == "R0"
  neg_r1 = SassRegister(name="R1", negated=True)
  assert str(neg_r1) == "-R1"
  abs_r2 = SassRegister(name="R2", absolute=True)
  assert str(abs_r2) == "|R2|"
  neg_abs_r3 = SassRegister(name="R3", negated=True, absolute=True)
  assert str(neg_abs_r3) == "-|R3|"


def test_predicate_formatting():
  """Verifies the behavior of predicate formatting."""
  p0 = SassPredicate(name="P0", is_guard=True)
  assert str(p0) == "@P0"
  not_p1 = SassPredicate(name="P1", negated=True, is_guard=True)
  assert str(not_p1) == "@!P1"


def test_immediate_formatting():
  """Verifies the behavior of immediate formatting."""
  i1 = SassImmediate(value=1)
  assert str(i1) == "1"
  f1 = SassImmediate(value=1.5)
  assert str(f1) == "1.5"
  h1 = SassImmediate(value=255, is_hex=True)
  assert str(h1) == "0xff"


def test_memory_formatting():
  """Verifies the behavior of memory formatting."""
  const_mem = SassMemory(base="c[0x0]", offset=4)
  assert str(const_mem) == "c[0x0][0x4]"
  r1 = SassRegister(name="R1")
  reg_mem = SassMemory(base=r1)
  assert str(reg_mem) == "[R1]"
  reg_mem_off = SassMemory(base=r1, offset=8)
  assert str(reg_mem_off) == "[R1 + 0x8]"


def test_instruction_formatting_basic():
  """Verifies the behavior of instruction formatting basic."""
  inst = SassInstruction(
    opcode="FADD", operands=[SassRegister(name="R0"), SassRegister(name="R1"), SassRegister(name="R2")]
  )
  assert str(inst) == "FADD R0, R1, R2;"


def test_instruction_with_predicate():
  """Verifies the behavior of instruction with predicate."""
  pred = SassPredicate(name="P0", negated=True, is_guard=True)
  inst = SassInstruction(opcode="MOV", operands=[SassRegister(name="R0"), SassRegister(name="RZ")], predicate=pred)
  assert str(inst) == "@!P0 MOV R0, RZ;"


def test_label_formatting():
  """Verifies the behavior of label formatting."""
  lbl = SassLabel(name="L_EXIT")
  assert str(lbl) == "L_EXIT:"


def test_directive_formatting():
  """Verifies the behavior of directive formatting."""
  d = SassDirective(name="headerflags", params=["@0x100"])
  assert str(d) == ".headerflags @0x100"


def test_comment_formatting():
  """Verifies the behavior of comment formatting."""
  c = SassComment(text="This is a loop")
  assert str(c) == "// This is a loop"
