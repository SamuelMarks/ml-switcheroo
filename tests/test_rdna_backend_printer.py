"""Test suite for rdna backend printer."""

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

  def to_text(self) -> str:
    """Execute to text helper logic.

    Returns:
        Resulting `str` from to text.
    """
    return "CustomNode"

  """Docstring."""

  def __str__(self) -> str:
    """Execute str helper logic.

    Returns:
        Resulting `str` from str  .
    """
    return "custom_node"


def test_rdna_printer() -> None:
  """Verify rdna printer behavior."""
  printer: RdnaPrinter = RdnaPrinter()
  nodes: list[RdnaNode] = [
    RdnaLabel(name="L1"),
    RdnaInstruction(opcode="v_mov_b32", operands=[RdnaImmediate(value=0)]),
    RdnaDirective(name="text"),
    RdnaComment(text="A comment"),
    CustomNode(),
  ]

  result: str = printer.emit(nodes)
  assert "L1:" in result and "v_mov_b32" in result
