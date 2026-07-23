"""Test suite for the Nodes module."""

import pytest
from ml_switcheroo.core.compiler.frontends.sass.nodes import (
  Register,
  Predicate,
  Immediate,
  Memory,
  Instruction,
  Label,
  Directive,
  Comment,
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
  assert str(Register("R0")) == "R0"
  assert str(Register("R1", negated=True)) == "-R1"
  assert str(Register("R2", absolute=True)) == "|R2|"
  assert str(Register("R3", negated=True, absolute=True)) == "-|R3|"


def test_predicate():
  """Verifies the behavior of predicate."""
  assert str(Predicate("P0")) == "P0"
  assert str(Predicate("P1", negated=True)) == "!P1"


def test_immediate():
  """Verifies the behavior of immediate."""
  assert str(Immediate(42)) == "42"
  assert str(Immediate(1.5)) == "1.5"
  assert str(Immediate(255, is_hex=True)) == "0xff"
  assert str(Immediate(1.5, is_hex=True)) == "0x1"


def test_memory():
  """Verifies the behavior of memory."""
  assert str(Memory("c[0x0]")) == "c[0x0][0x0]"
  assert str(Memory("c[0x0]", offset=4)) == "c[0x0][0x4]"
  assert str(Memory(Register("R1"))) == "[R1]"
  assert str(Memory(Register("R2"), offset=8)) == "[R2 + 0x8]"


def test_instruction():
  """Verifies the behavior of instruction."""
  assert str(Instruction("NOP")) == "NOP ;"
  assert str(Instruction("FADD", [Register("R0"), Register("R1")])) == "FADD R0, R1;"
  assert str(Instruction("FADD", [Register("R0"), Register("R1")], predicate=Predicate("P0"))) == "@P0 FADD R0, R1;"


def test_instruction_invalid():
  """Verifies the behavior of instruction invalid."""
  with pytest.raises(ValueError):
    Instruction("FADD R0")


def test_label():
  """Verifies the behavior of label."""
  assert str(Label("L_1")) == "L_1:"


def test_directive():
  """Verifies the behavior of directive."""
  assert str(Directive("headerflags")) == ".headerflags"
  assert str(Directive("global_base", ["foo", "bar"])) == ".global_base foo, bar"


def test_comment():
  """Verifies the behavior of comment."""
  assert str(Comment("hello")) == "// hello"
