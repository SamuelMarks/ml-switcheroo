"""Test suite for Phase 7: Other Operations in StableHLO.

Provides comprehensive coverage of exact semantics, type enforcement,
and generation correctness for remaining operations defined in the plan.
"""

import pytest
from pydantic import JsonValue

from ml_switcheroo.core.compiler.backends.stablehlo import StableHloBackend
from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode
from ml_switcheroo.semantics.manager import SemanticsManager


@pytest.fixture
def backend() -> StableHloBackend:
  """Provides a StableHLO Backend with a loaded SemanticsManager.

  Returns:
      Configured `backend` fixture value (StableHloBackend).
  """
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


@pytest.mark.parametrize(("logical_op", "expected_mlir_op"), OTHER_OPS)
def test_other_operations(backend: StableHloBackend, logical_op: str, expected_mlir_op: str) -> None:
  """Verifies that other operations are correctly mapped to StableHLO syntax.

  This ensures both mapping resolution and operand generation are correct.

  Args:
      backend: Input `backend` parameter value (`StableHloBackend`).
      logical_op: Input `logical_op` parameter value (`str`).
      expected_mlir_op: Input `expected_mlir_op` parameter value (`str`).
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
  """Verify `LogicalGraph` emits `stablehlo.custom_call` for stablehlo semantics not found."""
  # Hit 75->83

  class SemanticsNoDef(SemanticsManager):
    """Semantics manager stub returning no definition."""

    def __init__(self) -> None:
      """Initialize without loading default knowledge base."""
      super().__init__(load_defaults=False)

    def get_definition(self, api_name: str) -> tuple[str, dict[str, JsonValue]] | None:
      """Return None for all operation lookups.

      Args:
          api_name: Concrete API path or abstract operation ID.

      Returns:
          Always `None`.
      """
      return None

  backend = StableHloBackend(SemanticsNoDef())
  g = LogicalGraph("Test")
  g.add_node(LogicalNode(id="n1", op_type="not_found"))
  res: str = backend.compile(g)
  assert "stablehlo.custom_call" in res


def test_stablehlo_semantics_no_api() -> None:
  """Verify `LogicalGraph` emits `stablehlo.custom_call` for stablehlo semantics no API."""
  # Hit 80->83

  class SemanticsNoApi(SemanticsManager):
    """Semantics manager stub returning a variant without an API name."""

    def __init__(self) -> None:
      """Initialize without loading default knowledge base."""
      super().__init__(load_defaults=False)

    def get_definition(self, api_name: str) -> tuple[str, dict[str, JsonValue]] | None:
      """Return an operation definition with an empty StableHLO variant.

      Args:
          api_name: Concrete API path or abstract operation ID.

      Returns:
          Tuple of operation ID and variant dictionary lacking an `api` key.
      """
      return ("abs", {"variants": {"stablehlo": {}}})

  backend = StableHloBackend(SemanticsNoApi())
  g = LogicalGraph("Test")
  g.add_node(LogicalNode(id="n1", op_type="not_found"))
  res: str = backend.compile(g)
  assert "stablehlo.custom_call" in res


def test_stablehlo_semantics_none() -> None:
  """Verify `StableHloBackend` emits `stablehlo.custom_call` for stablehlo semantics none."""
  # Hit 75->83 (self.semantics is None)
  backend = StableHloBackend()
  backend.semantics = None
  g = LogicalGraph("Test")
  g.add_node(LogicalNode(id="n1", op_type="not_found"))
  res: str = backend.compile(g)
  assert "stablehlo.custom_call" in res
