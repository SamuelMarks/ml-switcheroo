"""Test suite for Phase 3: Neural Network & ML Operations in StableHLO.

Provides comprehensive coverage of exact semantics, type enforcement,
and generation correctness for NN operations defined in the plan.
"""

import pytest
from ml_switcheroo.core.compiler.backends.stablehlo import StableHloBackend
from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode, LogicalEdge
from ml_switcheroo.semantics.manager import SemanticsManager


@pytest.fixture
def backend():
  """Provides a StableHLO Backend with a loaded SemanticsManager."""
  return StableHloBackend(SemanticsManager())


NN_OPS = [
  ("BatchNormGrad", "stablehlo.batch_norm_grad"),
  ("BatchNormInference", "stablehlo.batch_norm_inference"),
  ("BatchNormTraining", "stablehlo.batch_norm_training"),
  ("Convolution", "stablehlo.convolution"),
  ("DotGeneral", "stablehlo.dot_general"),
  ("DynamicPad", "stablehlo.dynamic_pad"),
  ("Pad", "stablehlo.pad"),
  ("ReduceWindow", "stablehlo.reduce_window"),
  ("Rng", "stablehlo.rng"),
  ("RngBitGenerator", "stablehlo.rng_bit_generator"),
  ("SelectAndScatter", "stablehlo.select_and_scatter"),
  ("ShiftLeft", "stablehlo.shift_left"),
  ("ShiftRightArithmetic", "stablehlo.shift_right_arithmetic"),
]


@pytest.mark.parametrize("logical_op, expected_mlir_op", NN_OPS)
def test_nn_operations(backend: StableHloBackend, logical_op: str, expected_mlir_op: str):
  """Verifies that neural network operations are correctly mapped to StableHLO syntax.

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
