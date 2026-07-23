"""Test suite for the RdnaPrinter module."""

from ml_switcheroo.core.compiler.frontends.rdna.nodes import (
  Comment,
  Directive,
  Instruction,
  Label,
  RdnaNode,
)
from ml_switcheroo.core.compiler.backends.rdna.printer import RdnaPrinter


class CustomRdnaNode(RdnaNode):
  """Custom generic node for testing fallback behavior."""

  def __str__(self) -> str:
    """Return mock representation."""
    return "CustomNode"


def test_rdna_printer_emit_label():
  """Verifies that Label nodes are printed flush-left."""
  node = Label(name="L_1")
  printer = RdnaPrinter()
  output = printer.emit([node])
  assert output == "L_1:\n"


def test_rdna_printer_emit_instruction():
  """Verifies that Instruction nodes are printed indented."""
  node = Instruction(opcode="v_add_f32")
  printer = RdnaPrinter()
  output = printer.emit([node])
  assert output == "    v_add_f32\n"


def test_rdna_printer_emit_directive():
  """Verifies that Directive nodes are printed indented."""
  node = Directive(name="text", params=[])
  printer = RdnaPrinter()
  output = printer.emit([node])
  assert output == "    .text\n"


def test_rdna_printer_emit_comment():
  """Verifies that Comment nodes are printed indented."""
  node = Comment(text="This is a test")
  printer = RdnaPrinter()
  output = printer.emit([node])
  assert output == "    ; This is a test\n"


def test_rdna_printer_emit_fallback():
  """Verifies that unhandled node types fallback to indented str(node)."""
  node = CustomRdnaNode()
  printer = RdnaPrinter()
  output = printer.emit([node])
  assert output == "    CustomNode\n"


def test_rdna_printer_emit_multiple():
  """Verifies that multiple nodes are joined correctly."""
  nodes = [Directive(name="text", params=[]), Label(name="L_1"), Instruction(opcode="v_add_f32")]
  printer = RdnaPrinter()
  output = printer.emit(nodes)
  expected = "    .text\nL_1:\n    v_add_f32\n"
  assert output == expected
