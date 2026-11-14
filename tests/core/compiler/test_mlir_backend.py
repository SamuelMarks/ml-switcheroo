"""Tests for the MLIR Compiler Backend."""

from ml_switcheroo.core.compiler.backends.mlir_backend import MlirBackend
from ml_switcheroo.core.compiler.backends.mlir_printer import MlirPrinter
from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode
from ml_switcheroo.core.mlir.cst import BlockNode, ModuleNode, OperationNode
from ml_switcheroo.semantics.manager import SemanticsManager


def test_mlir_backend_init() -> None:
  """Verify MLIR backend init produces the expected state."""
  semantics = SemanticsManager(load_defaults=False)
  backend = MlirBackend(semantics=semantics)
  assert backend.semantics is semantics


def test_mlir_backend_compile_input_numeric() -> None:
  """Verify `MlirBackend` handles MLIR backend compile input numeric as expected."""
  backend = MlirBackend()
  graph = LogicalGraph(name="test")
  n1 = LogicalNode(id="n1", op_type="Input", attributes={"value": "42"})
  graph.add_node(n1)

  mlir_str: str = backend.compile(graph)
  assert "sw.constant" in mlir_str
  assert "value = 42" in mlir_str
  assert "i32" in mlir_str
  assert "%n1" in mlir_str


def test_mlir_backend_compile_input_non_numeric() -> None:
  """Verify `MlirBackend` handles MLIR backend compile input non numeric as expected."""
  backend = MlirBackend()
  graph = LogicalGraph(name="test")
  n1 = LogicalNode(id="n1", op_type="Input", attributes={"value": "not_a_number"})
  graph.add_node(n1)

  mlir_str: str = backend.compile(graph)
  assert "sw.op" in mlir_str
  assert 'type = "Input"' in mlir_str
  assert "!sw.unknown" in mlir_str


def test_mlir_backend_compile_input_default_numeric() -> None:
  """Verify `MlirBackend` handles MLIR backend compile input default numeric as expected."""
  backend = MlirBackend()
  graph = LogicalGraph(name="test")
  n1 = LogicalNode(id="n1", op_type="Input")  # metadata missing "value", should default to "1"
  graph.add_node(n1)

  mlir_str: str = backend.compile(graph)
  assert "sw.constant" in mlir_str
  assert "value = 1" in mlir_str


def test_mlir_backend_compile_output() -> None:
  """Verify `MlirBackend` emits `sw.return` for MLIR backend compile output."""
  backend = MlirBackend()
  graph = LogicalGraph(name="test")
  n1 = LogicalNode(id="n1", op_type="Output")
  graph.add_node(n1)

  mlir_str: str = backend.compile(graph)
  assert "sw.return" in mlir_str
  assert "()" in mlir_str


def test_mlir_backend_compile_generic_op() -> None:
  """Verify `MlirBackend` handles MLIR backend compile generic operation as expected."""
  backend = MlirBackend()
  graph = LogicalGraph(name="test")
  n1 = LogicalNode(id="n1", op_type="MyOp", attributes={"attr1": "val1", "attr2": "val2"})
  graph.add_node(n1)

  mlir_str: str = backend.compile(graph)
  assert "sw.op" in mlir_str
  assert 'type = "MyOp"' in mlir_str
  assert 'attr1 = "val1"' in mlir_str
  assert 'attr2 = "val2"' in mlir_str
  assert "%n1" in mlir_str


def test_mlir_printer_emit_non_module() -> None:
  """Verify `MlirPrinter` emits `mock.op` for MLIR printer emit non module."""
  printer = MlirPrinter()
  # OperationNode.to_text() should be called
  op = OperationNode(name='"mock.op"')
  text: str = printer.emit(op)
  assert "mock.op" in text


def test_mlir_printer_emit_module_with_module_op() -> None:
  """Verify `MlirPrinter` handles MLIR printer emit module with module operation as expected."""
  printer = MlirPrinter()
  op = OperationNode(name="module")
  block = BlockNode(label="")
  block.operations.append(op)
  module = ModuleNode(body=block)

  text: str = printer.emit(module, header="// Header\n")
  assert "// Header" in text
  assert "module" in text
  assert "func.func @main()" not in text


def test_mlir_printer_emit_module_without_module_op() -> None:
  """Verify `MlirPrinter` handles MLIR printer emit module without module operation as expected."""
  printer = MlirPrinter()
  op = OperationNode(name='"some.op"')
  block = BlockNode(label="")
  block.operations.append(op)
  module = ModuleNode(body=block)

  text: str = printer.emit(module, header="")
  assert "module {" in text
  assert "func.func @main() {" in text
  assert "some.op" in text


def test_mlir_backend_validate_op() -> None:
  """Test validate_op on MlirBackend."""
  from unittest.mock import patch

  backend = MlirBackend()
  report = backend.validate_op("arith.addi", operand_count=2)
  assert report is not None

  report_builtin = backend.validate_op("func", operand_count=0)
  assert report_builtin is not None

  with patch("ml_ecosystem_snapshots.grounding.compiler.validate_mlir_op", side_effect=RuntimeError("Failed")):
    assert backend.validate_op("arith.addi") is None
