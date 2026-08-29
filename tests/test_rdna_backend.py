"""Docstring."""

from typing import List

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
  """Docstring."""

  def to_text(self) -> str:
    """Docstring."""
    return "dummy"


def test_rdna_printer() -> None:
  """Docstring."""
  printer: RdnaPrinter = RdnaPrinter()
  nodes: List[RdnaNode] = [
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
