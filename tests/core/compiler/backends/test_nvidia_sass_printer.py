"""Unit tests for `NvidiaSassPrinter` (`ml_switcheroo.core.compiler.backends.nvidia_sass.printer`)."""

from ml_switcheroo.core.compiler.backends.nvidia_sass.printer import NvidiaSassPrinter
from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import (
  NvidiaSassComment,
  NvidiaSassDirective,
  NvidiaSassInstruction,
  NvidiaSassLabel,
  NvidiaSassNode,
  NvidiaSassRegister,
)

_NL = chr(10)


class _CustomSassNode(NvidiaSassNode):
  """Custom `NvidiaSassNode` exercising fallback `emit` formatting."""

  def to_text(self) -> str:
    """Return `'CustomNode'`.

    Returns:
        Literal `'CustomNode'`.
    """
    return "CustomNode"


def test_nvidia_sass_printer_labels_instructions_directives_comments_and_fallback() -> None:
  """Verify `NvidiaSassPrinter.emit` formats labels, instructions, directives, comments, and custom nodes."""
  printer = NvidiaSassPrinter()
  assert printer.emit([NvidiaSassLabel(name="L_1")]) == f"L_1:{_NL}"
  assert printer.emit([NvidiaSassInstruction(opcode="FADD")]).lower() == f"    fadd;{_NL}"
  assert ".headerflags" in printer.emit([NvidiaSassDirective(name="headerflags", params=[])])
  assert "This is a test" in printer.emit([NvidiaSassComment(text="This is a test")])
  assert printer.emit([_CustomSassNode()]) == f"CustomNode{_NL}"

  mixed: list[NvidiaSassNode] = [
    NvidiaSassLabel(name="L1"),
    NvidiaSassInstruction(opcode="MOV", operands=[NvidiaSassRegister("R0")]),
    NvidiaSassDirective(name=".global", params=["main"]),
    NvidiaSassComment(text="// test"),
    NvidiaSassRegister("R0"),
  ]
  out = printer.emit(mixed)
  assert "L1:" in out
  assert "mov" in out.lower()
  assert "..global main" in out
  assert "// // test" in out
  assert "R0" in out
