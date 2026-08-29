"""Test module."""

from typing import Any, List

from ml_switcheroo.core.compiler.backends.sass.printer import SassPrinter
from ml_switcheroo.core.compiler.frontends.sass.cst import (
  SassComment,
  SassDirective,
  SassInstruction,
  SassLabel,
  SassNode,
  SassRegister,
)


class MockSassNode(SassNode):
  """Docstring."""

  def __str__(self) -> str:
    """Docstring."""
    return "mock_node"

  def __eq__(self, other: Any) -> bool:
    """Docstring."""
    return isinstance(other, MockSassNode)


def test_sass_printer_emit() -> None:
  """Docstring."""
  printer: SassPrinter = SassPrinter()
  nodes: List[SassNode] = [
    SassLabel(name="L1"),
    SassInstruction(opcode="MOV", operands=[SassRegister(name="R0"), SassRegister(name="R1")]),
    SassDirective(name="version", params=["1.0"]),
    SassComment(text="comment"),
    MockSassNode(),
  ]

  result: str = printer.emit(nodes)

  expected: str = "L1:\n    MOV R0, R1;\n    .version 1.0\n    // comment\n    mock_node\n"
  assert result == expected
