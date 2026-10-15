"""Test suite for rdna backend printer."""

from typing import List

from ml_switcheroo.core.compiler.backends.rdna.printer import RdnaPrinter
from ml_switcheroo.core.compiler.frontends.rdna.cst import (
  RdnaComment,
  RdnaDirective,
  RdnaImmediate,
  RdnaInstruction,
  RdnaLabel,
  RdnaNode,
)


class CustomNode(RdnaNode):
  """Test helper class CustomNode."""

  def to_text(self):
    """Execute to text helper logic."""
    return "CustomNode"

  """Docstring."""

  def __str__(self) -> str:
    """Execute str helper logic."""
    return "custom_node"


def test_rdna_printer() -> None:
  """Verify rdna printer behavior."""
  printer: RdnaPrinter = RdnaPrinter()
  nodes: List[RdnaNode] = [
    RdnaLabel(name="L1"),
    RdnaInstruction(opcode="v_mov_b32", operands=[RdnaImmediate(value=0)]),
    RdnaDirective(name="text"),
    RdnaComment(text="A comment"),
    CustomNode(),
  ]

  result: str = printer.emit(nodes)
  assert "L1:" in result and "v_mov_b32" in result
