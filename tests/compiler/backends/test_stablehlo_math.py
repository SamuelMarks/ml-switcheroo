"""Test suite for Phase 2: Core Math Operations in StableHLO.

Provides comprehensive coverage of exact semantics, type enforcement,
and generation correctness for math operations defined in the plan.
"""

import pytest
from ml_switcheroo.core.compiler.backends.stablehlo import StableHloBackend
from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode, LogicalEdge
from ml_switcheroo.semantics.manager import SemanticsManager


@pytest.fixture
def backend():
  """Provides a StableHLO Backend with a loaded SemanticsManager."""
  return StableHloBackend(SemanticsManager())


MATH_OPS = [
  ("Abs", "stablehlo.abs"),
  ("Add", "stablehlo.add"),
  ("Atan2", "stablehlo.atan2"),
  ("Cbrt", "stablehlo.cbrt"),
  ("Ceil", "stablehlo.ceil"),
  ("Cholesky", "stablehlo.cholesky"),
  ("Clamp", "stablehlo.clamp"),
  ("Complex", "stablehlo.complex"),
  ("Cosine", "stablehlo.cosine"),
  ("Div", "stablehlo.divide"),
  ("Floor", "stablehlo.floor"),
  ("Imag", "stablehlo.imag"),
  ("IsFinite", "stablehlo.is_finite"),
  ("Log", "stablehlo.log"),
  ("LogPlusOne", "stablehlo.log_plus_one"),
  ("Logistic", "stablehlo.logistic"),
  ("Maximum", "stablehlo.maximum"),
  ("Minimum", "stablehlo.minimum"),
  ("Mul", "stablehlo.multiply"),
  ("Neg", "stablehlo.negate"),
  ("Pow", "stablehlo.power"),
  ("Real", "stablehlo.real"),
  ("Remainder", "stablehlo.remainder"),
  ("RoundNearestAfz", "stablehlo.round_nearest_afz"),
  ("RoundNearestEven", "stablehlo.round_nearest_even"),
  ("Rsqrt", "stablehlo.rsqrt"),
  ("ShiftRightLogical", "stablehlo.shift_right_logical"),
  ("Sign", "stablehlo.sign"),
  ("Sine", "stablehlo.sine"),
  ("Sqrt", "stablehlo.sqrt"),
  ("Sub", "stablehlo.subtract"),
  ("Tanh", "stablehlo.tanh"),
]


@pytest.mark.parametrize("logical_op, expected_mlir_op", MATH_OPS)
def test_math_operations(backend: StableHloBackend, logical_op: str, expected_mlir_op: str):
  """Verifies that mathematical operations are correctly mapped to StableHLO syntax.

  This ensures both mapping resolution and operand generation are correct.
  """
  g = LogicalGraph()
  # Simple graph: Input -> Op -> Output
  g.nodes = [LogicalNode("in_node", "Input"), LogicalNode("op_node", logical_op), LogicalNode("out_node", "Output")]
  g.edges = [LogicalEdge("in_node", "op_node"), LogicalEdge("op_node", "out_node")]

  mlir_code = backend.compile(g)

  # 1. Operation exists in MLIR output
  assert expected_mlir_op in mlir_code

  # 2. Input/Output structure
  assert "stablehlo.constant" in mlir_code
  assert "return" in mlir_code
  assert "%op_node =" in mlir_code


def test_stablehlo_custom_call_fallback(backend: StableHloBackend):
  """Verifies that unknown operations fall back to a custom_call."""
  g = LogicalGraph()
  g.nodes = [LogicalNode("my_op", "UnknownMagicOp")]
  mlir_code = backend.compile(g)
  assert "stablehlo.custom_call" in mlir_code
  assert "@unknownmagicop" in mlir_code
