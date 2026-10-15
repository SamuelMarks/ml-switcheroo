"""Test suite for rdna printer."""

from ml_switcheroo.core.compiler.backends.rdna.printer import RdnaPrinter
from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaComment
from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaComment as Comment
from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaDirective
from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaDirective as Directive
from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaInstruction
from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaInstruction as Instruction
from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaLabel
from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaLabel as Label
from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaNode


class CustomRdnaNode(RdnaNode):
  """Test helper class CustomRdnaNode."""

  def to_text(self) -> str:
    """Execute to text helper logic."""
    return "CustomNode"


def test_rdna_printer_emit_label() -> None:
  """Verifies that Label nodes are printed flush-left."""
  node = Label(name="L_1")
  printer = RdnaPrinter()
  output: str = printer.emit([node])
  assert output == "L_1:\n"


def test_rdna_printer_emit_instruction() -> None:
  """Verifies that Instruction nodes are printed indented."""
  node = Instruction(opcode="v_add_f32")
  printer = RdnaPrinter()
  output: str = printer.emit([node])
  assert output.strip().startswith("v_add_f32")


def test_rdna_printer_emit_directive() -> None:
  """Verifies that Directive nodes are printed indented."""
  node = Directive(name="text", params=[])
  printer = RdnaPrinter()
  output: str = printer.emit([node])
  assert output.strip().startswith(".text")


def test_rdna_printer_emit_comment() -> None:
  """Verifies that Comment nodes are printed indented."""
  node = Comment(text="This is a test")
  printer = RdnaPrinter()
  output: str = printer.emit([node])
  assert "This is a test" in output


def test_rdna_printer_emit_fallback() -> None:
  """Verifies that unhandled node types fallback to indented str(node)."""
  node = CustomRdnaNode()
  printer = RdnaPrinter()
  output: str = printer.emit([node])
  assert output == "CustomNode\n"


def test_rdna_printer_emit_multiple() -> None:
  """Verifies that multiple nodes are joined correctly."""
  nodes: list[RdnaNode] = [Directive(name="text", params=[]), Label(name="L_1"), Instruction(opcode="v_add_f32")]
  printer = RdnaPrinter()
  output: str = printer.emit(nodes)
  assert ".text" in output and "v_add_f32" in output


def test_rdna_printer_all_nodes() -> None:
  """Verify rdna printer all nodes behavior."""
  from ml_switcheroo.core.compiler.backends.rdna.printer import RdnaPrinter
  from ml_switcheroo.core.compiler.frontends.rdna.cst import c_SGPR

  printer = RdnaPrinter()
  nodes: list[RdnaNode] = [
    RdnaLabel(name="L1"),
    RdnaInstruction(opcode="v_add_f32", operands=[c_SGPR(0)]),
    RdnaDirective(name=".global", params=["main"]),
    RdnaComment(text="; test"),
    c_SGPR(0),  # test fallback
  ]
  txt: str = printer.emit(nodes)
  assert "L1:" in txt
  assert "v_add_f32" in txt
  assert ".global main" in txt
  assert "; test" in txt
  assert "s0" in txt
