"""Docstring."""

import typing


def test_rdna_cst_repr() -> None:
  """Docstring."""
  from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaNode
  from ml_switcheroo.core.cst.base import Trivia

  class DummyRdna(RdnaNode):
    def _get_name(self) -> str:
      return "Dummy"

    def _get_fields(self) -> dict[str, typing.Any]:
      return {"a": 1}

  d = DummyRdna()
  assert "Dummy" in repr(d)
  d.leading_trivia = [Trivia(text=" ")]  # type: ignore
  assert "leading_trivia" in repr(d)


def test_rdna_cst_repr_methods() -> None:
  """Docstring."""
  from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaDirective, RdnaMemory, RdnaSGPR, RdnaVGPR

  d = RdnaDirective(name="a", params=["1"])
  assert "a" in repr(d)

  m = RdnaMemory(base=RdnaSGPR(0), offset=4)
  assert "4" in repr(m)

  s = RdnaSGPR(index=0)
  assert "index=0" in repr(s)

  v = RdnaVGPR(index=0, count=2)
  assert "count=2" in repr(v)


def test_rdna_cst_render() -> None:
  """Docstring."""
  from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaLabelRef, RdnaModifier, c_SGPR, c_VGPR
  from ml_switcheroo.core.cst.base import Trivia

  label = RdnaLabelRef(name="label")
  label.leading_trivia = [Trivia(text=" ")]  # type: ignore
  label.trailing_trivia = [Trivia(text=" ")]  # type: ignore
  assert label.to_text() == " label "

  m = RdnaModifier(name="mod")
  m.leading_trivia = [Trivia(text=" ")]  # type: ignore
  m.trailing_trivia = [Trivia(text=" ")]  # type: ignore
  assert m.to_text() == " mod "

  s_reg: typing.Any = c_SGPR(1)
  assert s_reg.index == 1

  v_reg: typing.Any = c_VGPR(2)
  assert v_reg.index == 2


def test_rdna_instruction_validation() -> None:
  """Docstring."""
  import pytest

  from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaInstruction

  with pytest.raises(ValueError):
    RdnaInstruction(opcode="invalid code")


def test_rdna_instruction_render_empty_operands() -> None:
  """Docstring."""
  from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaImmediate, RdnaInstruction

  i = RdnaInstruction(opcode="nop", operands=[RdnaImmediate(value=0), RdnaImmediate(value=1)])  # type: ignore
  assert "nop 0, 1" in i.to_text()

  i2 = RdnaInstruction(opcode="nop")
  assert "nop" in i2.to_text()


def test_rdna_instruction_trivia() -> None:
  """Docstring."""
  from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaImmediate, RdnaInstruction, RdnaModule
  from ml_switcheroo.core.cst.base import Trivia

  op = RdnaImmediate(value=0)  # type: ignore
  op.leading_trivia = [Trivia(text=" /* trivia */ ")]  # type: ignore
  i = RdnaInstruction(opcode="nop", operands=[op])
  assert "/* trivia */" in i.to_text()

  mod = RdnaModule(statements=[i])  # type: ignore
  assert "/* trivia */" in mod.to_text()
