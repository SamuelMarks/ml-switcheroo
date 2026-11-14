"""Test suite for stablehlo backend."""

from pydantic import JsonValue

from ml_switcheroo.core.compiler.backends.mlir_printer import MlirPrinter
from ml_switcheroo.core.compiler.backends.stablehlo import StableHloBackend
from ml_switcheroo.core.compiler.ir import LogicalEdge, LogicalGraph, LogicalNode
from ml_switcheroo.core.mlir.cst import BlockNode, MlirNode, ModuleNode, OperationNode
from ml_switcheroo.semantics.manager import SemanticsManager


def test_stablehlo_backend() -> None:
  """Verify `LogicalGraph` emits `stablehlo.constant` for stablehlo backend."""

  class MockSemantics(SemanticsManager):
    """Test double `MockSemantics` implementing `get_definition` for stablehlo backend tests."""

    def __init__(self) -> None:
      """Initialize empty semantics manager without disk hydration."""
      super().__init__(load_defaults=False)

    def get_definition(self, api_name: str) -> tuple[str, dict[str, JsonValue]] | None:
      """Execute `get_definition` on `MockSemantics` for stablehlo backend tests.

      Args:
          api_name: Operation name to resolve.

      Returns:
          Resulting `tuple[str, dict[str, JsonValue]] | None` from get definition.
      """
      if api_name == "KnownOp":
        return ("abstract.KnownOp", {"variants": {"stablehlo": {"api": "stablehlo.add"}}})
      return None

  backend = StableHloBackend(semantics=MockSemantics())

  graph = LogicalGraph(
    nodes={
      n.id: n
      for n in [
        LogicalNode(id="in1", op_type="Input"),
        LogicalNode(id="unkn1", op_type="UnknownOp"),
        LogicalNode(id="known1", op_type="KnownOp"),
        LogicalNode(id="out1", op_type="Output"),
      ]
    },
    edges=[LogicalEdge("in1", "unkn1"), LogicalEdge("unkn1", "known1"), LogicalEdge("known1", "out1")],
  )

  code = backend.compile(graph)
  assert "stablehlo.constant" in code
  assert "return" in code
  assert "stablehlo.custom_call" in code
  assert "call_target_name" in code
  assert "stablehlo.add" in code


def test_stablehlo_backend_no_semantics() -> None:
  """Verify `StableHloBackend` emits `stablehlo.custom_call` for stablehlo backend no semantics."""
  backend = StableHloBackend(semantics=None)
  graph = LogicalGraph(nodes={n.id: n for n in [LogicalNode(id="node1", op_type="MyOp")]}, edges=[])
  code = backend.compile(graph)
  assert "stablehlo.custom_call" in code


def test_mlir_printer() -> None:
  """Verify `MlirPrinter` emits `// Header` for MLIR printer."""
  printer = MlirPrinter()
  module = ModuleNode(body=BlockNode(label=""))
  module.body.operations.append(OperationNode(name="dummy.op"))
  code = printer.emit(module, header="// Header\n")
  assert "// Header" in code
  assert "module" in code
  assert "dummy.op" in code

  # testing _visit coverage for fallback
  class DummyNode(MlirNode):
    """Test double `DummyNode` implementing `to_text` for stablehlo backend tests."""

    def to_text(self) -> str:
      """Execute `to_text` on `DummyNode` for stablehlo backend tests.

      Returns:
          Resulting `str` from to text.
      """
      return "dummy_text"

  code2 = printer.emit(DummyNode())
  assert "dummy_text" in code2


def test_stablehlo_backend_validate_op() -> None:
  """Test validate_op on StableHloBackend."""
  from unittest.mock import patch

  backend = StableHloBackend()
  report = backend.validate_op("add")
  assert report is not None

  with patch("ml_ecosystem_snapshots.grounding.compiler.validate_stablehlo_op", side_effect=RuntimeError("Error")):
    assert backend.validate_op("add") is None


def test_mlir_printer_with_module_op() -> None:
  """Verify `MlirPrinter` handles MLIR printer with module operation as expected."""
  printer = MlirPrinter()
  module = ModuleNode(body=BlockNode(label=""))
  module.body.operations.append(OperationNode(name="module"))
  code = printer.emit(module, header="")
  assert "module" in code
