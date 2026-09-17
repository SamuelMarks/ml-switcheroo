"""Test suite for Phase 7: Other Operations in StableHLO.

Provides comprehensive coverage of exact semantics, type enforcement,
and generation correctness for remaining operations defined in the plan.
"""

import typing

import pytest

from ml_switcheroo.core.compiler.backends.stablehlo import StableHloBackend
from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode
from ml_switcheroo.semantics.manager import SemanticsManager


@pytest.fixture
def backend() -> StableHloBackend:
  """Provides a StableHLO Backend with a loaded SemanticsManager."""
  return StableHloBackend(SemanticsManager())


OTHER_OPS: list[tuple[str, str]] = [
  ("AfterAll", "stablehlo.after_all"),
  ("And", "stablehlo.and"),
  ("AsyncDone", "stablehlo.async_done"),
  ("AsyncStart", "stablehlo.async_start"),
  ("Compare", "stablehlo.compare"),
  ("Composite", "stablehlo.composite"),
  ("CountLeadingZeros", "stablehlo.count_leading_zeros"),
  ("Exponential", "stablehlo.exponential"),
  ("ExponentialMinusOne", "stablehlo.exponential_minus_one"),
  ("Fft", "stablehlo.fft"),
  ("GetDimensionSize", "stablehlo.get_dimension_size"),
  ("GetTupleElement", "stablehlo.get_tuple_element"),
  ("Infeed", "stablehlo.infeed"),
  ("Map", "stablehlo.map"),
  ("Not", "stablehlo.not"),
  ("Or", "stablehlo.or"),
  ("Outfeed", "stablehlo.outfeed"),
  ("Popcnt", "stablehlo.popcnt"),
  ("Reduce", "stablehlo.reduce"),
  ("ReducePrecision", "stablehlo.reduce_precision"),
  ("Select", "stablehlo.select"),
  ("Tan", "stablehlo.tan"),
  ("TriangularSolve", "stablehlo.triangular_solve"),
  ("Tuple", "stablehlo.tuple"),
  ("Xor", "stablehlo.xor"),
]


@pytest.mark.parametrize("logical_op, expected_mlir_op", OTHER_OPS)
def test_other_operations(backend: StableHloBackend, logical_op: str, expected_mlir_op: str) -> None:
  """Verifies that other operations are correctly mapped to StableHLO syntax.

  This ensures both mapping resolution and operand generation are correct.
  """
  in_n = LogicalNode(id="in_node", op_type="Input")
  op_n = LogicalNode(id="op_node", op_type=logical_op, inputs=["in_node"])
  out_n = LogicalNode(id="out_node", op_type="Output", inputs=["op_node"])
  g = LogicalGraph(nodes={"in_node": in_n, "op_node": op_n, "out_node": out_n})

  mlir_code: str = backend.compile(g)

  # 1. Operation exists in MLIR output
  assert expected_mlir_op in mlir_code

  # 2. Input/Output structure
  assert "stablehlo.constant" in mlir_code
  assert "return" in mlir_code
  assert "%op_node =" in mlir_code


def test_stablehlo_semantics_not_found() -> None:
  """Docstring."""
  # Hit 75->83

  class SemanticsNoDef:
    """Semantics no def."""

    def get_definition(self, kind: str) -> typing.Optional[tuple[str, dict[str, typing.Any]]]:
      """Get definition."""
      return None

  backend = StableHloBackend(SemanticsNoDef())  # type: ignore
  g = LogicalGraph("Test")
  g.add_node(LogicalNode(id="n1", op_type="not_found"))
  res: str = backend.compile(g)
  assert "stablehlo.custom_call" in res


def test_stablehlo_semantics_no_api() -> None:
  """Docstring."""
  # Hit 80->83

  class SemanticsNoApi:
    """Semantics no api."""

    def get_definition(self, kind: str) -> typing.Optional[tuple[str, dict[str, typing.Any]]]:
      """Get definition."""
      return ("abs", {"variants": {"stablehlo": {}}})

  backend = StableHloBackend(SemanticsNoApi())  # type: ignore
  g = LogicalGraph("Test")
  g.add_node(LogicalNode(id="n1", op_type="not_found"))
  res: str = backend.compile(g)
  assert "stablehlo.custom_call" in res


def test_stablehlo_semantics_none() -> None:
  """Docstring."""
  # Hit 75->83 (self.semantics is None)
  backend = StableHloBackend()
  backend.semantics = None
  g = LogicalGraph("Test")
  g.add_node(LogicalNode(id="n1", op_type="not_found"))
  res: str = backend.compile(g)
  assert "stablehlo.custom_call" in res
