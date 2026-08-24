"""Docstring."""

from ml_switcheroo.core.compiler.backends.stablehlo import StableHloBackend
from ml_switcheroo.core.compiler.backends.mlir_printer import MlirPrinter
from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode, LogicalEdge
from ml_switcheroo.core.mlir.cst import ModuleNode, BlockNode, OperationNode


def test_stablehlo_backend():
  """Docstring."""

  class MockSemantics:
    """Docstring."""

    def get_definition(self, kind):
      """Docstring."""
      if kind == "KnownOp":
        return ("abstract.KnownOp", {"variants": {"stablehlo": {"api": "stablehlo.add"}}})
      return None

  backend = StableHloBackend(semantics=MockSemantics())

  graph = LogicalGraph(
    nodes=[
      LogicalNode(id="in1", kind="Input"),
      LogicalNode(id="unkn1", kind="UnknownOp"),
      LogicalNode(id="known1", kind="KnownOp"),
      LogicalNode(id="out1", kind="Output"),
    ],
    edges=[LogicalEdge("in1", "unkn1"), LogicalEdge("unkn1", "known1"), LogicalEdge("known1", "out1")],
  )

  code = backend.compile(graph)
  assert "stablehlo.constant" in code
  assert "return" in code
  assert "stablehlo.custom_call" in code
  assert "call_target_name" in code
  assert "stablehlo.add" in code


def test_stablehlo_backend_no_semantics():
  """Docstring."""
  backend = StableHloBackend(semantics=None)
  graph = LogicalGraph(nodes=[LogicalNode(id="node1", kind="MyOp")], edges=[])
  code = backend.compile(graph)
  assert "stablehlo.custom_call" in code


def test_mlir_printer():
  """Docstring."""
  printer = MlirPrinter()
  module = ModuleNode(body=BlockNode(label=""))
  module.body.operations.append(OperationNode(name="dummy.op"))
  code = printer.emit(module, header="// Header\n")
  assert "// Header" in code
  assert "module" in code
  assert "dummy.op" in code

  # testing _visit coverage for fallback
  class DummyNode:
    """Docstring."""

    def to_text(self):
      """Docstring."""
      return "dummy_text"

  code2 = printer.emit(DummyNode())
  assert "dummy_text" in code2


def test_mlir_printer_with_module_op():
  """Docstring."""
  printer = MlirPrinter()
  module = ModuleNode(body=BlockNode(label=""))
  module.body.operations.append(OperationNode(name="module"))
  code = printer.emit(module, header="")
  assert "module" in code
