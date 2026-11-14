"""Test suite for rdna backend."""

from ml_switcheroo.core.compiler.backends.rdna.printer import RdnaPrinter
from ml_switcheroo.core.compiler.frontends.rdna.cst import (
  RdnaComment,
  RdnaDirective,
  RdnaInstruction,
  RdnaLabel,
  RdnaNode,
  RdnaVGPR,
)


class DummyRdnaNode(RdnaNode):
  """Stub `RdnaNode` subclass implementing `to_text` for RDNA backend tests."""

  def to_text(self) -> str:
    """Execute `to_text` on `DummyRdnaNode` for RDNA backend tests.

    Returns:
        Resulting `str` from to text.
    """
    return "dummy"


def test_rdna_printer() -> None:
  """Verify `RdnaPrinter` emits `label1` for RDNA printer."""
  printer: RdnaPrinter = RdnaPrinter()
  nodes: list[RdnaNode] = [
    RdnaLabel(name="label1"),
    RdnaInstruction(opcode="v_add_f32", operands=[RdnaVGPR(index=0), RdnaVGPR(index=1)]),
    RdnaDirective(name=".global", params=[]),
    RdnaComment(text="comment"),
    DummyRdnaNode(),  # fallback
  ]

  out: str = printer.emit(nodes)

  assert "label1" in out
  assert "v_add_f32" in out
  assert ".global" in out
  assert "comment" in out
  assert "dummy" in out
