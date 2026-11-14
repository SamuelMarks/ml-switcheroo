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
  """Stub `RdnaNode` subclass implementing `to_text` for RDNA backend printer tests."""

  def to_text(self) -> str:
    """Execute `to_text` on `CustomNode` for RDNA backend printer tests.

    Returns:
        Resulting `str` from to text.
    """
    return "CustomNode"

  """Docstring."""

  def __str__(self) -> str:
    """Execute `__str__` on `CustomNode` for RDNA backend printer tests.

    Returns:
        Resulting `str` from str  .
    """
    return "custom_node"


def test_rdna_printer() -> None:
  """Verify `RdnaPrinter` emits `L1:` for RDNA printer."""
  printer: RdnaPrinter = RdnaPrinter()
  nodes: list[RdnaNode] = [
    RdnaLabel(name="L1"),
    RdnaInstruction(opcode="v_mov_b32", operands=[RdnaImmediate(value=0)]),
    RdnaDirective(name="text"),
    RdnaComment(text="A comment"),
    CustomNode(),
  ]

  result: str = printer.emit(nodes)
  assert "L1:" in result
  assert "v_mov_b32" in result
