"""Test suite for the MlirPrinter module."""

from ml_switcheroo.core.mlir.cst import ModuleNode, BlockNode, OperationNode, ValueNode, AttributeNode, TypeNode
from ml_switcheroo.core.compiler.backends.mlir_printer import MlirPrinter


def test_mlir_printer_emit_module_wrapper():
  """Verifies that the printer emits a wrapper when there is no explicit module op."""
  block = BlockNode(label="")
  op = OperationNode(
    name='"sw.op"',
    results=[ValueNode(name="%out")],
    attributes=[AttributeNode("type", '"Input"')],
    result_types=[TypeNode("!sw.unknown")],
  )
  block.operations.append(op)
  module = ModuleNode(body=block)

  printer = MlirPrinter()
  output = printer.emit(module)

  assert "// Graph -> MLIR compilation output" in output
  assert "module {" in output
  assert "func.func @main() {" in output
  assert '%out = "sw.op" {type = "Input"} : !sw.unknown' in output
  assert "}" in output


def test_mlir_printer_emit_explicit_module():
  """Verifies that the printer does not duplicate module wrappers."""
  block = BlockNode(label="")
  module_op = OperationNode(
    name="module",
  )
  block.operations.append(module_op)
  module = ModuleNode(body=block)

  printer = MlirPrinter()
  output = printer.emit(module)

  assert "// Graph -> MLIR compilation output" in output
  assert output.count("module {\n") == 0  # Should just use to_text() which omits braces for simple op


def test_mlir_printer_emit_non_module():
  """Verifies that emitting a non-module node delegates to to_text()."""
  op = OperationNode(name='"sw.return"', result_types=[TypeNode("()")])

  printer = MlirPrinter()
  output = printer.emit(op)

  assert '"sw.return" : ()' in output
