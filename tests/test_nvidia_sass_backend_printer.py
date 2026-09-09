"""Test module."""

from typing import Any, List

from ml_switcheroo.core.compiler.backends.nvidia_sass.printer import NvidiaSassPrinter
from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import (
  NvidiaSassComment,
  NvidiaSassDirective,
  NvidiaSassInstruction,
  NvidiaSassLabel,
  NvidiaSassNode,
  NvidiaSassRegister,
)


class MockNvidiaSassNode(NvidiaSassNode):
  """Docstring."""

  def __str__(self) -> str:
    """Docstring."""
    return "mock_node"

  def __eq__(self, other: Any) -> bool:
    """Docstring."""
    return isinstance(other, MockNvidiaSassNode)


def test_nvidia_sass_printer_emit() -> None:
  """Docstring."""
  printer: NvidiaSassPrinter = NvidiaSassPrinter()
  nodes: List[NvidiaSassNode] = [
    NvidiaSassLabel(name="L1"),
    NvidiaSassInstruction(opcode="MOV", operands=[NvidiaSassRegister(name="R0"), NvidiaSassRegister(name="R1")]),
    NvidiaSassDirective(name="version", params=["1.0"]),
    NvidiaSassComment(text="comment"),
    MockNvidiaSassNode(),
  ]

  result: str = printer.emit(nodes)

  expected: str = "L1:\n    MOV R0, R1;\n    .version 1.0\n    // comment\n    mock_node\n"
  assert result == expected
