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
from typing import List


class CustomNode(RdnaNode):
  """Test element."""

  def __str__(self) -> str:
    """Test element."""
    return "custom_node"


def test_rdna_printer() -> None:
  """Test element."""
  printer: RdnaPrinter = RdnaPrinter()
  nodes: List[RdnaNode] = [
    RdnaLabel(name="L1"),
    RdnaInstruction(opcode="v_mov_b32", operands=[RdnaImmediate(value=0)]),
    RdnaDirective(name="text"),
    RdnaComment(text="A comment"),
    CustomNode(),
  ]

  result: str = printer.emit(nodes)
  expected: str = "L1:\n    v_mov_b32 0\n    .text\n    ; A comment\n    custom_node\n"
  assert result == expected
