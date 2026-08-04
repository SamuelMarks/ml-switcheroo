"""Test suite for the Nodes module."""

import pytest
from ml_switcheroo.core.compiler.frontends.sass.cst import (
  SassRegister,
  SassPredicate,
  SassImmediate,
  SassMemory,
  SassInstruction,
  SassLabel,
  SassDirective,
  SassComment,
  SassNode,
)


def test_operand_base():
  """Verifies the behavior of operand base."""

  class DummyNode(SassNode):
    """Dummy."""

    def __str__(self):
      """Str."""
      return "dummy"

  d = DummyNode()
  assert str(d) == "dummy"


def test_register():
  """Verifies the behavior of register."""
  assert str(SassRegister(name="R0")) == "R0"
  assert str(SassRegister(name="R1", negated=True)) == "-R1"
  assert str(SassRegister(name="R2", absolute=True)) == "|R2|"
  assert str(SassRegister(name="R3", negated=True, absolute=True)) == "-|R3|"


def test_predicate():
  """Verifies the behavior of predicate."""
  assert str(SassPredicate(name="P0")) == "P0"
  assert str(SassPredicate(name="P1", negated=True)) == "!P1"


def test_immediate():
  """Verifies the behavior of immediate."""
  assert str(SassImmediate(value=42)) == "42"
  assert str(SassImmediate(value=1.5)) == "1.5"
  assert str(SassImmediate(value=255, is_hex=True)) == "0xff"
  assert str(SassImmediate(value=1.5, is_hex=True)) == "0x1"


def test_memory():
  """Verifies the behavior of memory."""
  assert str(SassMemory(base="c[0x0]")) == "c[0x0][0x0]"
  assert str(SassMemory(base="c[0x0]", offset=4)) == "c[0x0][0x4]"
  assert str(SassMemory(base=SassRegister(name="R1"))) == "[R1]"
  assert str(SassMemory(base=SassRegister(name="R2"), offset=8)) == "[R2 + 0x8]"


def test_instruction():
  """Verifies the behavior of instruction."""
  assert str(SassInstruction(opcode="NOP")) == "NOP;"
  assert (
    str(SassInstruction(opcode="FADD", operands=[SassRegister(name="R0"), SassRegister(name="R1")])) == "FADD R0, R1;"
  )
  assert (
    str(
      SassInstruction(
        opcode="FADD",
        operands=[SassRegister(name="R0"), SassRegister(name="R1")],
        predicate=SassPredicate(name="P0", is_guard=True),
      )
    )
    == "@P0 FADD R0, R1;"
  )


def test_instruction_invalid():
  """Verifies the behavior of instruction invalid."""
  with pytest.raises(ValueError):
    SassInstruction(opcode="FADD R0")


def test_label():
  """Verifies the behavior of label."""
  assert str(SassLabel(name="L_1")) == "L_1:"


def test_directive():
  """Verifies the behavior of directive."""
  assert str(SassDirective(name="headerflags")) == ".headerflags"
  assert str(SassDirective(name="global_base", params=["foo", "bar"])) == ".global_base foo, bar"


def test_comment():
  """Verifies the behavior of comment."""
  assert str(SassComment(text="hello")) == "// hello"
