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


class DummySassNode(SassNode):
  """Docstring."""

  def to_text(self):
    """Docstring."""
    return "dummy"


def test_sass_printer():
  """Docstring."""
  printer = SassPrinter()
  nodes = [
    SassLabel(name="label1"),
    SassInstruction(opcode="FADD", operands=[SassRegister(name="R0"), SassRegister(name="R1")]),
    SassDirective(name=".global", params=[]),
    SassComment(text="comment"),
    DummySassNode(),  # fallback
  ]

  out = printer.emit(nodes)

  assert "label1" in out
  assert "FADD" in out
  assert ".global" in out
  assert "comment" in out
  assert "dummy" in out
