"""Test suite for the SassPrinter module."""

from ml_switcheroo.core.compiler.backends.sass.printer import SassPrinter
from ml_switcheroo.core.compiler.frontends.sass.cst import SassComment as Comment
from ml_switcheroo.core.compiler.frontends.sass.cst import SassDirective as Directive
from ml_switcheroo.core.compiler.frontends.sass.cst import SassInstruction as Instruction
from ml_switcheroo.core.compiler.frontends.sass.cst import SassLabel as Label
from ml_switcheroo.core.compiler.frontends.sass.cst import SassNode


class CustomSassNode(SassNode):
  """Docstring."""

  def __str__(self) -> str:
    """Return mock representation."""
    return "CustomNode"


def test_sass_printer_emit_label() -> None:
  """Verifies that Label nodes are printed flush-left."""
  node = Label(name="L_1")
  printer = SassPrinter()
  output: str = printer.emit([node])
  assert output == "L_1:\n"


def test_sass_printer_emit_instruction() -> None:
  """Verifies that Instruction nodes are printed indented."""
  node = Instruction(opcode="FADD")
  printer = SassPrinter()
  output: str = printer.emit([node])
  assert output == "    FADD;\n"


def test_sass_printer_emit_directive() -> None:
  """Verifies that Directive nodes are printed indented."""
  node = Directive(name="headerflags", params=[])
  printer = SassPrinter()
  output: str = printer.emit([node])
  assert output == "    .headerflags\n"


def test_sass_printer_emit_comment() -> None:
  """Verifies that Comment nodes are printed indented."""
  node = Comment(text="This is a test")
  printer = SassPrinter()
  output: str = printer.emit([node])
  assert output == "    // This is a test\n"


def test_sass_printer_emit_fallback() -> None:
  """Verifies that unhandled node types fallback to indented str(node)."""
  node = CustomSassNode()
  printer = SassPrinter()
  output: str = printer.emit([node])
  assert output == "    CustomNode\n"


def test_sass_printer_emit_multiple() -> None:
  """Verifies that multiple nodes are joined correctly."""
  nodes: list[SassNode] = [Directive(name="headerflags", params=[]), Label(name="L_1"), Instruction(opcode="FADD")]
  printer = SassPrinter()
  output: str = printer.emit(nodes)
  expected: str = "    .headerflags\nL_1:\n    FADD;\n"
  assert output == expected


def test_sass_printer_all_nodes() -> None:
  """Docstring."""
  from ml_switcheroo.core.compiler.backends.sass.printer import SassPrinter
  from ml_switcheroo.core.compiler.frontends.sass.cst import (
    SassComment,
    SassDirective,
    SassInstruction,
    SassLabel,
    SassRegister,
  )

  printer = SassPrinter()
  nodes: list[SassNode] = [
    SassLabel(name="L1"),
    SassInstruction(opcode="MOV", operands=[SassRegister("R0")]),
    SassDirective(name=".global", params=["main"]),
    SassComment(text="// test"),
    SassRegister("R0"),  # test fallback
  ]
  txt: str = printer.emit(nodes)
  assert "L1:" in txt
  assert "MOVR0" in txt
  assert "..global main" in txt
  assert "// // test" in txt
  assert "R0" in txt
