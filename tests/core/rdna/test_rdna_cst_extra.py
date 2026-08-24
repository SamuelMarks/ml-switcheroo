"""Docstring."""


def test_rdna_cst_repr():
  """Docstring."""
  from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaNode
  from ml_switcheroo.core.cst.base import Trivia

  class DummyRdna(RdnaNode):
    """Docstring."""

    def _get_name(self):
      """Docstring."""
      return "Dummy"

    def _get_fields(self):
      """Docstring."""
      return {"a": 1}

  d = DummyRdna()
  assert "Dummy" in repr(d)
  d.leading_trivia = [Trivia(text=" ")]
  assert "leading_trivia" in repr(d)


def test_rdna_cst_repr_methods():
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


def test_rdna_cst_render():
  """Docstring."""
  from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaLabelRef, RdnaModifier, c_SGPR, c_VGPR
  from ml_switcheroo.core.cst.base import Trivia

  label = RdnaLabelRef(name="label")
  label.leading_trivia = [Trivia(text=" ")]
  label.trailing_trivia = [Trivia(text=" ")]
  assert label.to_text() == " label "

  m = RdnaModifier(name="mod")
  m.leading_trivia = [Trivia(text=" ")]
  m.trailing_trivia = [Trivia(text=" ")]
  assert m.to_text() == " mod "

  s_reg = c_SGPR(1)
  assert s_reg.index == 1

  v_reg = c_VGPR(2)
  assert v_reg.index == 2


def test_rdna_instruction_validation():
  """Docstring."""
  from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaInstruction
  import pytest

  with pytest.raises(ValueError):
    RdnaInstruction(opcode="invalid code")


def test_rdna_instruction_render_empty_operands():
  """Docstring."""
  from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaInstruction, RdnaImmediate

  i = RdnaInstruction(opcode="nop", operands=[RdnaImmediate(value=0), RdnaImmediate(value=1)])
  assert "nop 0, 1" in i.to_text()

  i2 = RdnaInstruction(opcode="nop")
  assert "nop" in i2.to_text()


def test_rdna_instruction_trivia():
  """Docstring."""
  from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaInstruction, RdnaImmediate, RdnaModule
  from ml_switcheroo.core.cst.base import Trivia

  op = RdnaImmediate(value=0)
  op.leading_trivia = [Trivia(text=" /* trivia */ ")]
  i = RdnaInstruction(opcode="nop", operands=[op])
  assert "/* trivia */" in i.to_text()

  mod = RdnaModule(statements=[i])
  assert "/* trivia */" in mod.to_text()
