"""Tests for the MLIR Compiler Backend."""

from ml_switcheroo.core.compiler.backends.mlir_backend import MlirBackend
from ml_switcheroo.core.compiler.backends.mlir_printer import MlirPrinter
from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode
from ml_switcheroo.core.mlir.cst import ModuleNode, BlockNode, OperationNode


def test_mlir_backend_init():
  """Test initialization."""
  backend = MlirBackend(semantics="mock_semantics")
  assert backend.semantics == "mock_semantics"


def test_mlir_backend_compile_input_numeric():
  """Test compile with numeric Input node."""
  backend = MlirBackend()
  graph = LogicalGraph(name="test")
  n1 = LogicalNode(id="n1", kind="Input", metadata={"value": "42"})
  graph.nodes.append(n1)

  mlir_str = backend.compile(graph)
  assert "sw.constant" in mlir_str
  assert "value = 42" in mlir_str
  assert "i32" in mlir_str
  assert "%n1" in mlir_str


def test_mlir_backend_compile_input_non_numeric():
  """Test compile with non-numeric Input node."""
  backend = MlirBackend()
  graph = LogicalGraph(name="test")
  n1 = LogicalNode(id="n1", kind="Input", metadata={"value": "not_a_number"})
  graph.nodes.append(n1)

  mlir_str = backend.compile(graph)
  assert "sw.op" in mlir_str
  assert 'type = "Input"' in mlir_str
  assert "!sw.unknown" in mlir_str


def test_mlir_backend_compile_input_default_numeric():
  """Test compile with Input node with no value in metadata (defaults to 1)."""
  backend = MlirBackend()
  graph = LogicalGraph(name="test")
  n1 = LogicalNode(id="n1", kind="Input")  # metadata missing "value", should default to "1"
  graph.nodes.append(n1)

  mlir_str = backend.compile(graph)
  assert "sw.constant" in mlir_str
  assert "value = 1" in mlir_str


def test_mlir_backend_compile_output():
  """Test compile with Output node."""
  backend = MlirBackend()
  graph = LogicalGraph(name="test")
  n1 = LogicalNode(id="n1", kind="Output")
  graph.nodes.append(n1)

  mlir_str = backend.compile(graph)
  assert "sw.return" in mlir_str
  assert "()" in mlir_str


def test_mlir_backend_compile_generic_op():
  """Test compile with a generic operation node."""
  backend = MlirBackend()
  graph = LogicalGraph(name="test")
  n1 = LogicalNode(id="n1", kind="MyOp", metadata={"attr1": "val1", "attr2": "val2"})
  graph.nodes.append(n1)

  mlir_str = backend.compile(graph)
  assert "sw.op" in mlir_str
  assert 'type = "MyOp"' in mlir_str
  assert 'attr1 = "val1"' in mlir_str
  assert 'attr2 = "val2"' in mlir_str
  assert "%n1" in mlir_str


def test_mlir_printer_emit_non_module():
  """Test MlirPrinter with non-module node."""
  printer = MlirPrinter()
  # OperationNode.to_text() should be called
  op = OperationNode(name='"mock.op"')
  text = printer.emit(op)
  assert "mock.op" in text


def test_mlir_printer_emit_module_with_module_op():
  """Test MlirPrinter with a module node that already has a module operation."""
  printer = MlirPrinter()
  op = OperationNode(name="module")
  block = BlockNode(label="")
  block.operations.append(op)
  module = ModuleNode(body=block)

  text = printer.emit(module, header="// Header\n")
  assert "// Header" in text
  assert "module" in text
  assert "func.func @main()" not in text


def test_mlir_printer_emit_module_without_module_op():
  """Test MlirPrinter with a module node that has normal ops."""
  printer = MlirPrinter()
  op = OperationNode(name='"some.op"')
  block = BlockNode(label="")
  block.operations.append(op)
  module = ModuleNode(body=block)

  text = printer.emit(module, header="")
  assert "module {" in text
  assert "func.func @main() {" in text
  assert "some.op" in text
