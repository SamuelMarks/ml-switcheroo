"""Test suite for the NvidiaSassPrinter module."""

from ml_switcheroo.core.compiler.backends.nvidia_sass.printer import NvidiaSassPrinter
from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassComment as Comment
from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassDirective as Directive
from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassInstruction as Instruction
from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassLabel as Label
from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassNode


class CustomNvidiaSassNode(NvidiaSassNode):
  """Docstring."""

  def __str__(self) -> str:
    """Return mock representation."""
    return "CustomNode"


def test_nvidia_sass_printer_emit_label() -> None:
  """Verifies that Label nodes are printed flush-left."""
  node = Label(name="L_1")
  printer = NvidiaSassPrinter()
  output: str = printer.emit([node])
  assert output == "L_1:\n"


def test_nvidia_sass_printer_emit_instruction() -> None:
  """Verifies that Instruction nodes are printed indented."""
  node = Instruction(opcode="FADD")
  printer = NvidiaSassPrinter()
  output: str = printer.emit([node])
  assert output == "    FADD;\n"


def test_nvidia_sass_printer_emit_directive() -> None:
  """Verifies that Directive nodes are printed indented."""
  node = Directive(name="headerflags", params=[])
  printer = NvidiaSassPrinter()
  output: str = printer.emit([node])
  assert output == "    .headerflags\n"


def test_nvidia_sass_printer_emit_comment() -> None:
  """Verifies that Comment nodes are printed indented."""
  node = Comment(text="This is a test")
  printer = NvidiaSassPrinter()
  output: str = printer.emit([node])
  assert output == "    // This is a test\n"


def test_nvidia_sass_printer_emit_fallback() -> None:
  """Verifies that unhandled node types fallback to indented str(node)."""
  node = CustomNvidiaSassNode()
  printer = NvidiaSassPrinter()
  output: str = printer.emit([node])
  assert output == "    CustomNode\n"


def test_nvidia_sass_printer_emit_multiple() -> None:
  """Verifies that multiple nodes are joined correctly."""
  nodes: list[NvidiaSassNode] = [Directive(name="headerflags", params=[]), Label(name="L_1"), Instruction(opcode="FADD")]
  printer = NvidiaSassPrinter()
  output: str = printer.emit(nodes)
  expected: str = "    .headerflags\nL_1:\n    FADD;\n"
  assert output == expected


def test_nvidia_sass_printer_all_nodes() -> None:
  """Docstring."""
  from ml_switcheroo.core.compiler.backends.nvidia_sass.printer import NvidiaSassPrinter
  from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import (
    NvidiaSassComment,
    NvidiaSassDirective,
    NvidiaSassInstruction,
    NvidiaSassLabel,
    NvidiaSassRegister,
  )

  printer = NvidiaSassPrinter()
  nodes: list[NvidiaSassNode] = [
    NvidiaSassLabel(name="L1"),
    NvidiaSassInstruction(opcode="MOV", operands=[NvidiaSassRegister("R0")]),
    NvidiaSassDirective(name=".global", params=["main"]),
    NvidiaSassComment(text="// test"),
    NvidiaSassRegister("R0"),  # test fallback
  ]
  txt: str = printer.emit(nodes)
  assert "L1:" in txt
  assert "MOVR0" in txt
  assert "..global main" in txt
  assert "// // test" in txt
  assert "R0" in txt
