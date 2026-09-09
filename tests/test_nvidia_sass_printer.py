"""Docstring."""

from typing import List

from ml_switcheroo.core.compiler.backends.nvidia_sass.printer import NvidiaSassPrinter
from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import (
  NvidiaSassComment,
  NvidiaSassDirective,
  NvidiaSassInstruction,
  NvidiaSassLabel,
  NvidiaSassNode,
  NvidiaSassRegister,
)


class DummyNvidiaSassNode(NvidiaSassNode):
  """Docstring."""

  def to_text(self) -> str:
    """Docstring."""
    return "dummy"


def test_nvidia_sass_printer() -> None:
  """Docstring."""
  printer: NvidiaSassPrinter = NvidiaSassPrinter()
  nodes: List[NvidiaSassNode] = [
    NvidiaSassLabel(name="label1"),
    NvidiaSassInstruction(opcode="FADD", operands=[NvidiaSassRegister(name="R0"), NvidiaSassRegister(name="R1")]),
    NvidiaSassDirective(name=".global", params=[]),
    NvidiaSassComment(text="comment"),
    DummyNvidiaSassNode(),  # fallback
  ]

  out: str = printer.emit(nodes)

  assert "label1" in out
  assert "FADD" in out
  assert ".global" in out
  assert "comment" in out
  assert "dummy" in out
