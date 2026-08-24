"""Test module."""

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
  """Test element."""

  def __str__(self):
    """Test element."""
    return "mock_node"

  def __eq__(self, other):
    """Test element."""
    return isinstance(other, MockSassNode)


def test_sass_printer_emit():
  """Test element."""
  printer = SassPrinter()
  nodes = [
    SassLabel(name="L1"),
    SassInstruction(opcode="MOV", operands=[SassRegister(name="R0"), SassRegister(name="R1")]),
    SassDirective(name="version", params=["1.0"]),
    SassComment(text="comment"),
    MockSassNode(),
  ]

  result = printer.emit(nodes)

  expected = "L1:\n" "    MOV R0, R1;\n" "    .version 1.0\n" "    // comment\n" "    mock_node\n"
  assert result == expected
