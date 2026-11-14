"""Test suite for nvidia sass backend printer."""

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
  """Fallback `NvidiaSassNode` test stub exercising `__str__` emission."""

  def to_text(self) -> str:
    """Render textual form of the mock node.

    Returns:
        Literal `'MockNode'` string.
    """
    return "MockNode"

  def __str__(self) -> str:
    """Render fallback string form of the mock node.

    Returns:
        Literal `'mock_node'` string.
    """
    return "mock_node"


def test_nvidia_sass_printer_emit() -> None:
  """Verify `NvidiaSassPrinter` emits `L1:` for nvidia SASS printer emit."""
  printer: NvidiaSassPrinter = NvidiaSassPrinter()
  nodes: list[NvidiaSassNode] = [
    NvidiaSassLabel(name="L1"),
    NvidiaSassInstruction(opcode="MOV", operands=[NvidiaSassRegister(name="R0"), NvidiaSassRegister(name="R1")]),
    NvidiaSassDirective(name="version", params=["1.0"]),
    NvidiaSassComment(text="comment"),
    MockNvidiaSassNode(),
  ]

  result: str = printer.emit(nodes)

  assert "L1:" in result
  assert "mov" in result
  assert ".version" in result
