"""Test suite for stablehlo backend."""

import typing
from pydantic import JsonValue


from ml_switcheroo.core.compiler.backends.mlir_printer import MlirPrinter
from ml_switcheroo.core.compiler.backends.stablehlo import StableHloBackend
from ml_switcheroo.core.compiler.ir import LogicalEdge, LogicalGraph, LogicalNode
from ml_switcheroo.core.mlir.cst import BlockNode, ModuleNode, OperationNode


def test_stablehlo_backend() -> None:
  """Verify stablehlo backend behavior."""

  class MockSemantics:
    """Test helper class MockSemantics."""

    def get_definition(self, kind: str) -> typing.Optional[tuple[str, dict[str, JsonValue]]]:
      """Execute get definition helper logic.

      Args:
          kind: Test argument or fixture.

      Returns:
          Computed test result or fixture value.
      """
      if kind == "KnownOp":
        return ("abstract.KnownOp", {"variants": {"stablehlo": {"api": "stablehlo.add"}}})
      return None

  backend = StableHloBackend(semantics=MockSemantics())  # type: ignore[arg-type]

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
  """Verify stablehlo backend no semantics behavior."""
  backend = StableHloBackend(semantics=None)
  graph = LogicalGraph(nodes={n.id: n for n in [LogicalNode(id="node1", op_type="MyOp")]}, edges=[])
  code = backend.compile(graph)
  assert "stablehlo.custom_call" in code


def test_mlir_printer() -> None:
  """Verify mlir printer behavior."""
  printer = MlirPrinter()
  module = ModuleNode(body=BlockNode(label=""))
  module.body.operations.append(OperationNode(name="dummy.op"))
  code = printer.emit(module, header="// Header\n")
  assert "// Header" in code
  assert "module" in code
  assert "dummy.op" in code

  # testing _visit coverage for fallback
  class DummyNode:
    """Test helper class DummyNode."""

    def to_text(self) -> str:
      """Execute to text helper logic.

      Returns:
          Computed test result or fixture value.
      """
      return "dummy_text"

  code2 = printer.emit(DummyNode())  # type: ignore[arg-type]
  assert "dummy_text" in code2


def test_stablehlo_backend_validate_op() -> None:
  """Test validate_op on StableHloBackend."""
  from unittest.mock import patch

  backend = StableHloBackend()
  report = backend.validate_op("add")
  assert report is not None

  with patch("ml_ecosystem_snapshots.grounding.compiler.validate_stablehlo_op", side_effect=Exception("Error")):
    assert backend.validate_op("add") is None


def test_mlir_printer_with_module_op() -> None:
  """Verify mlir printer with module op behavior."""
  printer = MlirPrinter()
  module = ModuleNode(body=BlockNode(label=""))
  module.body.operations.append(OperationNode(name="module"))
  code = printer.emit(module, header="")
  assert "module" in code
