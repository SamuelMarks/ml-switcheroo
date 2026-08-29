"""Docstring."""

import pytest


def test_sass_cst_render_edge() -> None:
  """Docstring."""
  from ml_switcheroo.core.compiler.frontends.sass.cst import SassLabel, SassMemory, SassPredicate, SassRegister
  from ml_switcheroo.core.cst.base import Trivia

  r = SassRegister(name="R0", negated=True, absolute=True)
  r.leading_trivia = [Trivia(text=" ")]  # type: ignore
  r.trailing_trivia = [Trivia(text=" ")]  # type: ignore
  assert r.to_text() == " -|R0| "

  m = SassMemory(base=SassRegister(name="R1"), offset=None)
  m.leading_trivia = [Trivia(text=" ")]  # type: ignore
  assert m.to_text() == " [R1]"

  m2 = SassMemory(base=SassRegister(name="R1"), offset=-16)
  assert m2.to_text() == "[R1 + -0x10]"

  p = SassPredicate(name="P0", negated=True, is_guard=True)
  p.leading_trivia = [Trivia(text=" ")]  # type: ignore
  assert p.to_text() == " @!P0"

  label = SassLabel(name="label")
  label.leading_trivia = [Trivia(text=" ")]  # type: ignore
  assert label.to_text() == " label:"

  cm = SassMemory(base="c[0x0]", offset=None)
  cm.leading_trivia = [Trivia(text=" ")]  # type: ignore
  assert cm.to_text() == " c[0x0][0x0]"

  cm2 = SassMemory(base="c[0x0]", offset=4)
  assert cm2.to_text() == "c[0x0][0x4]"


def test_sass_instruction_cst() -> None:
  """Docstring."""
  from ml_switcheroo.core.compiler.frontends.sass.cst import SassImmediate, SassInstruction, SassPredicate

  with pytest.raises(ValueError):
    SassInstruction(opcode="invalid op")

  i = SassInstruction(opcode="NOP", operands=[SassImmediate(value=0), SassImmediate(value=1)])  # type: ignore
  assert "NOP 0, 1" in i.to_text()

  i2 = SassInstruction(opcode="NOP")
  assert "NOP" in i2.to_text()

  p = SassPredicate(name="P0", is_guard=True)
  i3 = SassInstruction(opcode="NOP", predicate=p)
  assert "@P0 NOP" in i3.to_text()


def test_sass_instruction_trivia() -> None:
  """Docstring."""
  from ml_switcheroo.core.compiler.frontends.sass.cst import SassImmediate, SassInstruction, SassModule
  from ml_switcheroo.core.cst.base import Trivia

  op = SassImmediate(value=0)  # type: ignore
  op.leading_trivia = [Trivia(text=" /* trivia */ ")]  # type: ignore
  i = SassInstruction(opcode="NOP", operands=[op])
  assert "/* trivia */" in i.to_text()

  mod = SassModule(statements=[i])  # type: ignore
  assert "/* trivia */" in mod.to_text()
