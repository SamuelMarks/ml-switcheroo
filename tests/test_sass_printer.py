"""Docstring."""

from ml_switcheroo.core.compiler.backends.sass.printer import SassPrinter
from ml_switcheroo.core.compiler.frontends.sass.cst import (
  SassComment,
  SassDirective,
  SassInstruction,
  SassLabel,
  SassNode,
  SassRegister,
)
from typing import List


class DummySassNode(SassNode):
  """Docstring."""

  def to_text(self) -> str:
    """Docstring."""
    return "dummy"


def test_sass_printer() -> None:
  """Docstring."""
  printer: SassPrinter = SassPrinter()
  nodes: List[SassNode] = [
    SassLabel(name="label1"),
    SassInstruction(opcode="FADD", operands=[SassRegister(name="R0"), SassRegister(name="R1")]),
    SassDirective(name=".global", params=[]),
    SassComment(text="comment"),
    DummySassNode(),  # fallback
  ]

  out: str = printer.emit(nodes)

  assert "label1" in out
  assert "FADD" in out
  assert ".global" in out
  assert "comment" in out
  assert "dummy" in out
