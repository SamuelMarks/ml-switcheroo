"""Test suite for nvidia sass printer."""

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
  """Test helper class DummyNvidiaSassNode."""

  def to_text(self) -> str:
    """Execute to text helper logic.

    Returns:
        Resulting `str` from to text.
    """
    return "dummy"


def test_nvidia_sass_printer() -> None:
  """Verify nvidia sass printer behavior."""
  printer: NvidiaSassPrinter = NvidiaSassPrinter()
  nodes: list[NvidiaSassNode] = [
    NvidiaSassLabel(name="label1"),
    NvidiaSassInstruction(opcode="FADD", operands=[NvidiaSassRegister(name="R0"), NvidiaSassRegister(name="R1")]),
    NvidiaSassDirective(name=".global", params=[]),
    NvidiaSassComment(text="comment"),
    DummyNvidiaSassNode(),  # fallback
  ]

  out: str = printer.emit(nodes)

  assert "label1" in out
  assert "fadd" in out.lower()
  assert ".global" in out
  assert "comment" in out
  assert "dummy" in out
