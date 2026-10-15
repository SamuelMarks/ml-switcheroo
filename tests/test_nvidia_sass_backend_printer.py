"""Test suite for nvidia sass backend printer."""

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
  """Test helper class MockNvidiaSassNode."""

  def to_text(self):
    """Execute to text helper logic."""
    return "MockNode"

  """Docstring."""

  def __str__(self) -> str:
    """Execute str helper logic."""
    return "mock_node"

  def __eq__(self, other: Any) -> bool:
    """Execute eq helper logic."""
    return isinstance(other, MockNvidiaSassNode)


def test_nvidia_sass_printer_emit() -> None:
  """Verify nvidia sass printer emit behavior."""
  printer: NvidiaSassPrinter = NvidiaSassPrinter()
  nodes: List[NvidiaSassNode] = [
    NvidiaSassLabel(name="L1"),
    NvidiaSassInstruction(opcode="MOV", operands=[NvidiaSassRegister(name="R0"), NvidiaSassRegister(name="R1")]),
    NvidiaSassDirective(name="version", params=["1.0"]),
    NvidiaSassComment(text="comment"),
    MockNvidiaSassNode(),
  ]

  result: str = printer.emit(nodes)

  assert "L1:" in result and "mov" in result and ".version" in result
