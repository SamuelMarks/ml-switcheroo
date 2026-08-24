"""Test module."""

from ml_switcheroo.core.compiler.frontends.rdna.cst import (
  RdnaComment,
  RdnaDirective,
  RdnaInstruction,
  RdnaLabel,
  RdnaNode,
  RdnaImmediate,
)
from ml_switcheroo.core.compiler.backends.rdna.printer import RdnaPrinter


class CustomNode(RdnaNode):
  """Test element."""

  def __str__(self):
    """Test element."""
    return "custom_node"


def test_rdna_printer():
  """Test element."""
  printer = RdnaPrinter()
  nodes = [
    RdnaLabel(name="L1"),
    RdnaInstruction(opcode="v_mov_b32", operands=[RdnaImmediate(value=0)]),
    RdnaDirective(name="text"),
    RdnaComment(text="A comment"),
    CustomNode(),
  ]

  result = printer.emit(nodes)
  expected = "L1:\n" "    v_mov_b32 0\n" "    .text\n" "    ; A comment\n" "    custom_node\n"
  assert result == expected
