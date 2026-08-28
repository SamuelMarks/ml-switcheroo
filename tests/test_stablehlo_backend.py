"""Docstring."""

from ml_switcheroo.core.compiler.backends.stablehlo import StableHloBackend
from ml_switcheroo.core.compiler.backends.mlir_printer import MlirPrinter
from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode, LogicalEdge
from ml_switcheroo.core.mlir.cst import ModuleNode, BlockNode, OperationNode
from typing import Dict, Any, Optional, Tuple


def test_stablehlo_backend() -> None:
  """Docstring."""

  class MockSemantics:
    """Docstring."""

    def get_definition(self, kind: str) -> Optional[Tuple[str, Dict[str, Any]]]:
      """Docstring."""
      if kind == "KnownOp":
        return ("abstract.KnownOp", {"variants": {"stablehlo": {"api": "stablehlo.add"}}})
      return None

  backend: StableHloBackend = StableHloBackend(semantics=MockSemantics())

  graph: LogicalGraph = LogicalGraph(
    nodes=[
      LogicalNode(id="in1", kind="Input"),
      LogicalNode(id="unkn1", kind="UnknownOp"),
      LogicalNode(id="known1", kind="KnownOp"),
      LogicalNode(id="out1", kind="Output"),
    ],
    edges=[LogicalEdge("in1", "unkn1"), LogicalEdge("unkn1", "known1"), LogicalEdge("known1", "out1")],
  )

  code: str = backend.compile(graph)
  assert "stablehlo.constant" in code
  assert "return" in code
  assert "stablehlo.custom_call" in code
  assert "call_target_name" in code
  assert "stablehlo.add" in code


def test_stablehlo_backend_no_semantics() -> None:
  """Docstring."""
  backend: StableHloBackend = StableHloBackend(semantics=None)
  graph: LogicalGraph = LogicalGraph(nodes=[LogicalNode(id="node1", kind="MyOp")], edges=[])
  code: str = backend.compile(graph)
  assert "stablehlo.custom_call" in code


def test_mlir_printer() -> None:
  """Docstring."""
  printer: MlirPrinter = MlirPrinter()
  module: ModuleNode = ModuleNode(body=BlockNode(label=""))
  module.body.operations.append(OperationNode(name="dummy.op"))
  code: str = printer.emit(module, header="// Header\n")
  assert "// Header" in code
  assert "module" in code
  assert "dummy.op" in code

  # testing _visit coverage for fallback
  class DummyNode:
    """Docstring."""

    def to_text(self) -> str:
      """Docstring."""
      return "dummy_text"

  code2: str = printer.emit(DummyNode())
  assert "dummy_text" in code2


def test_mlir_printer_with_module_op() -> None:
  """Docstring."""
  printer: MlirPrinter = MlirPrinter()
  module: ModuleNode = ModuleNode(body=BlockNode(label=""))
  module.body.operations.append(OperationNode(name="module"))
  code: str = printer.emit(module, header="")
  assert "module" in code
