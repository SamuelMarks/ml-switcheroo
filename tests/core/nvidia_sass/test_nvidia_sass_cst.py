"""Docstring."""

import pytest


def test_nvidia_sass_cst_render_edge() -> None:
  """Docstring."""
  from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import (
    NvidiaSassLabel,
    NvidiaSassMemory,
    NvidiaSassPredicate,
    NvidiaSassRegister,
  )
  from ml_switcheroo.core.cst.base import Trivia

  r = NvidiaSassRegister(name="R0", negated=True, absolute=True)
  r.leading_trivia = [Trivia(text=" ")]  # type: ignore
  r.trailing_trivia = [Trivia(text=" ")]  # type: ignore
  assert r.to_text() == " -|R0| "

  m = NvidiaSassMemory(base=NvidiaSassRegister(name="R1"), offset=None)
  m.leading_trivia = [Trivia(text=" ")]  # type: ignore
  assert m.to_text() == " [R1]"

  m2 = NvidiaSassMemory(base=NvidiaSassRegister(name="R1"), offset=-16)
  assert m2.to_text() == "[R1 + -0x10]"

  p = NvidiaSassPredicate(name="P0", negated=True, is_guard=True)
  p.leading_trivia = [Trivia(text=" ")]  # type: ignore
  assert p.to_text() == " @!P0"

  label = NvidiaSassLabel(name="label")
  label.leading_trivia = [Trivia(text=" ")]  # type: ignore
  assert label.to_text() == " label:"

  cm = NvidiaSassMemory(base="c[0x0]", offset=None)
  cm.leading_trivia = [Trivia(text=" ")]  # type: ignore
  assert cm.to_text() == " c[0x0][0x0]"

  cm2 = NvidiaSassMemory(base="c[0x0]", offset=4)
  assert cm2.to_text() == "c[0x0][0x4]"


def test_nvidia_sass_instruction_cst() -> None:
  """Docstring."""
  from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import (
    NvidiaSassImmediate,
    NvidiaSassInstruction,
    NvidiaSassPredicate,
  )

  with pytest.raises(ValueError):
    NvidiaSassInstruction(opcode="invalid op")

  i = NvidiaSassInstruction(opcode="NOP", operands=[NvidiaSassImmediate(value=0), NvidiaSassImmediate(value=1)])  # type: ignore
  assert "NOP 0, 1" in i.to_text()

  i2 = NvidiaSassInstruction(opcode="NOP")
  assert "NOP" in i2.to_text()

  p = NvidiaSassPredicate(name="P0", is_guard=True)
  i3 = NvidiaSassInstruction(opcode="NOP", predicate=p)
  assert "@P0 NOP" in i3.to_text()


def test_nvidia_sass_instruction_trivia() -> None:
  """Docstring."""
  from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import (
    NvidiaSassImmediate,
    NvidiaSassInstruction,
    NvidiaSassModule,
  )
  from ml_switcheroo.core.cst.base import Trivia

  op = NvidiaSassImmediate(value=0)  # type: ignore
  op.leading_trivia = [Trivia(text=" /* trivia */ ")]  # type: ignore
  i = NvidiaSassInstruction(opcode="NOP", operands=[op])
  assert "/* trivia */" in i.to_text()

  mod = NvidiaSassModule(statements=[i])  # type: ignore
  assert "/* trivia */" in mod.to_text()
