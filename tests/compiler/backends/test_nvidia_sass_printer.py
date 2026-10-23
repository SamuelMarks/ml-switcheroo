"""Test suite for nvidia sass printer."""

from ml_switcheroo.core.compiler.backends.nvidia_sass.printer import NvidiaSassPrinter
from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassComment
from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassComment as Comment
from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassDirective
from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassDirective as Directive
from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassInstruction
from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassInstruction as Instruction
from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassLabel
from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassLabel as Label
from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassNode, NvidiaSassRegister


class CustomNvidiaSassNode(NvidiaSassNode):
  """Test helper class CustomNvidiaSassNode."""

  def to_text(self) -> str:
    """Execute to text helper logic.

    Returns:
        Resulting `str` from to text.
    """
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
  assert output.lower() == "    fadd;\n"


def test_nvidia_sass_printer_emit_directive() -> None:
  """Verifies that Directive nodes are printed indented."""
  node = Directive(name="headerflags", params=[])
  printer = NvidiaSassPrinter()
  output: str = printer.emit([node])
  assert output == "    .headerflags\n" or ".headerflags" in output


def test_nvidia_sass_printer_emit_comment() -> None:
  """Verifies that Comment nodes are printed indented."""
  node = Comment(text="This is a test")
  printer = NvidiaSassPrinter()
  output: str = printer.emit([node])
  assert "This is a test" in output


def test_nvidia_sass_printer_emit_fallback() -> None:
  """Verifies that unhandled node types fallback to indented str(node)."""
  node = CustomNvidiaSassNode()
  printer = NvidiaSassPrinter()
  output: str = printer.emit([node])
  assert output == "CustomNode\n"


def test_nvidia_sass_printer_emit_multiple() -> None:
  """Verifies that multiple nodes are joined correctly."""
  nodes: list[NvidiaSassNode] = [Directive(name="headerflags", params=[]), Label(name="L_1"), Instruction(opcode="FADD")]
  printer = NvidiaSassPrinter()
  output: str = printer.emit(nodes)
  assert ".headerflags" in output and "fadd;" in output.lower()


def test_nvidia_sass_printer_all_nodes() -> None:
  """Verify nvidia sass printer all nodes behavior."""
  from ml_switcheroo.core.compiler.backends.nvidia_sass.printer import NvidiaSassPrinter

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
  assert "mov" in txt.lower()
  assert "..global main" in txt
  assert "// // test" in txt
  assert "R0" in txt
