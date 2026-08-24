"""Test suite for the SassPrinter module."""

from ml_switcheroo.core.compiler.frontends.sass.cst import (
  SassComment as Comment,
  SassDirective as Directive,
  SassInstruction as Instruction,
  SassLabel as Label,
  SassNode,
)
from ml_switcheroo.core.compiler.backends.sass.printer import SassPrinter


class CustomSassNode(SassNode):
  """Custom generic node for testing fallback behavior."""

  def __str__(self) -> str:
    """Return mock representation."""
    return "CustomNode"


def test_sass_printer_emit_label():
  """Verifies that Label nodes are printed flush-left."""
  node = Label(name="L_1")
  printer = SassPrinter()
  output = printer.emit([node])
  assert output == "L_1:\n"


def test_sass_printer_emit_instruction():
  """Verifies that Instruction nodes are printed indented."""
  node = Instruction(opcode="FADD")
  printer = SassPrinter()
  output = printer.emit([node])
  assert output == "    FADD;\n"


def test_sass_printer_emit_directive():
  """Verifies that Directive nodes are printed indented."""
  node = Directive(name="headerflags", params=[])
  printer = SassPrinter()
  output = printer.emit([node])
  assert output == "    .headerflags\n"


def test_sass_printer_emit_comment():
  """Verifies that Comment nodes are printed indented."""
  node = Comment(text="This is a test")
  printer = SassPrinter()
  output = printer.emit([node])
  assert output == "    // This is a test\n"


def test_sass_printer_emit_fallback():
  """Verifies that unhandled node types fallback to indented str(node)."""
  node = CustomSassNode()
  printer = SassPrinter()
  output = printer.emit([node])
  assert output == "    CustomNode\n"


def test_sass_printer_emit_multiple():
  """Verifies that multiple nodes are joined correctly."""
  nodes = [Directive(name="headerflags", params=[]), Label(name="L_1"), Instruction(opcode="FADD")]
  printer = SassPrinter()
  output = printer.emit(nodes)
  expected = "    .headerflags\nL_1:\n    FADD;\n"
  assert output == expected


def test_sass_printer_all_nodes():
  """Docstring."""
  from ml_switcheroo.core.compiler.backends.sass.printer import SassPrinter
  from ml_switcheroo.core.compiler.frontends.sass.cst import (
    SassLabel,
    SassInstruction,
    SassDirective,
    SassComment,
    SassRegister,
  )

  printer = SassPrinter()
  nodes = [
    SassLabel(name="L1"),
    SassInstruction(opcode="MOV", operands=[SassRegister("R0")]),
    SassDirective(name=".global", params=["main"]),
    SassComment(text="// test"),
    SassRegister("R0"),  # test fallback
  ]
  txt = printer.emit(nodes)
  assert "L1:" in txt
  assert "MOVR0" in txt
  assert "..global main" in txt
  assert "// // test" in txt
  assert "R0" in txt
