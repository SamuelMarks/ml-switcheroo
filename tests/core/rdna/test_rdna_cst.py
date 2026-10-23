"""Test suite for rdna cst."""

from pydantic import JsonValue

from ml_switcheroo.core.compiler.frontends.rdna.cst import (
  RdnaDirective,
  RdnaImmediate,
  RdnaInstruction,
  RdnaLabelRef,
  RdnaMemory,
  RdnaModifier,
  RdnaSGPR,
  RdnaVGPR,
)


def test_rdna_cst_repr() -> None:
  """Verify rdna cst repr behavior."""
  from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaNode
  from ml_switcheroo.core.cst.base import Trivia

  class DummyRdna(RdnaNode):
    """Test helper class DummyRdna."""

    def _get_name(self) -> str:
      """Execute get name helper logic.

      Returns:
          Resulting `str` from get name.
      """
      return "Dummy"

    def _get_fields(self) -> dict[str, JsonValue]:
      """Execute get fields helper logic.

      Returns:
          Resulting `dict[str, JsonValue]` from get fields.
      """
      return {"a": 1}

  d = DummyRdna()
  assert "Dummy" in repr(d)
  d.leading_trivia = [Trivia(text=" ")]
  assert "leading_trivia" in repr(d)


def test_rdna_cst_repr_methods() -> None:
  """Verify rdna cst repr methods behavior."""
  d = RdnaDirective(name="a", params=["1"])
  assert "a" in repr(d)

  m = RdnaMemory(base=RdnaSGPR(0), offset=4)
  assert "4" in repr(m)

  s = RdnaSGPR(index=0)
  assert "index=0" in repr(s)

  v = RdnaVGPR(index=0, count=2)
  assert "count=2" in repr(v)


def test_rdna_cst_render() -> None:
  """Verify rdna cst render behavior."""
  from ml_switcheroo.core.compiler.frontends.rdna.cst import c_SGPR, c_VGPR
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


def test_rdna_instruction_validation() -> None:
  """Verify rdna instruction validation behavior."""
  import pytest

  from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaInstruction

  with pytest.raises(ValueError):
    RdnaInstruction(opcode="invalid code")


def test_rdna_instruction_render_empty_operands() -> None:
  """Verify rdna instruction render empty operands behavior."""
  from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaInstruction

  i = RdnaInstruction(opcode="nop", operands=[RdnaImmediate(value=0), RdnaImmediate(value=1)])
  assert "nop 0, 1" in i.to_text()

  i2 = RdnaInstruction(opcode="nop")
  assert "nop" in i2.to_text()


def test_rdna_instruction_trivia() -> None:
  """Verify rdna instruction trivia behavior."""
  from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaModule
  from ml_switcheroo.core.cst.base import Trivia

  op = RdnaImmediate(value=0)
  op.leading_trivia = [Trivia(text=" /* trivia */ ")]
  i = RdnaInstruction(opcode="nop", operands=[op])
  assert "/* trivia */" in i.to_text()

  mod = RdnaModule(statements=[i])
  assert "/* trivia */" in mod.to_text()
