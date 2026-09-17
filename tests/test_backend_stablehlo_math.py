"""Test suite for Math & Arithmetic operations in the StableHLO backend."""

from typing import Any, Dict, List, Optional, Tuple

import libcst as cst
import pytest

from ml_switcheroo.core.compiler.backends.stablehlo import StableHloBackend
from ml_switcheroo.core.compiler.ir import LogicalEdge, LogicalGraph, LogicalNode
from ml_switcheroo.core.mlir.stablehlo_emitter import StableHloEmitter
from ml_switcheroo.semantics.manager import SemanticsManager

MATH_OPS: List[Tuple[str, str]] = [
  ("Abs", "stablehlo.abs"),
  ("Add", "stablehlo.add"),
  ("Atan2", "stablehlo.atan2"),
  ("Cbrt", "stablehlo.cbrt"),
  ("Ceil", "stablehlo.ceil"),
  ("Cosine", "stablehlo.cosine"),
  ("Div", "stablehlo.divide"),
  ("Exponential", "stablehlo.exponential"),
  ("ExponentialMinusOne", "stablehlo.exponential_minus_one"),
  ("Floor", "stablehlo.floor"),
  ("Log", "stablehlo.log"),
  ("LogPlusOne", "stablehlo.log_plus_one"),
  ("Logistic", "stablehlo.logistic"),
  ("Maximum", "stablehlo.maximum"),
  ("Minimum", "stablehlo.minimum"),
  ("Mul", "stablehlo.multiply"),
  ("Negate", "stablehlo.negate"),
  ("Power", "stablehlo.power"),
  ("Remainder", "stablehlo.remainder"),
  ("RoundNearestAfz", "stablehlo.round_nearest_afz"),
  ("RoundNearestEven", "stablehlo.round_nearest_even"),
  ("Rsqrt", "stablehlo.rsqrt"),
  ("Sign", "stablehlo.sign"),
  ("Sine", "stablehlo.sine"),
  ("Sqrt", "stablehlo.sqrt"),
  ("Subtract", "stablehlo.subtract"),
  ("Tan", "stablehlo.tan"),
  ("Tanh", "stablehlo.tanh"),
]


class MathSemanticsMock:
  """Mock semantics manager for math ops."""

  def get_definition(self, kind: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Mock lookup.

    Args:
        kind (str): Kind string.

    Returns:
        Optional[Tuple[str, Dict[str, Any]]]: Definition tuple.
    """
    for abstract, stablehlo_api in MATH_OPS:
      if kind == abstract:
        return abstract, {"variants": {"stablehlo": {"api": stablehlo_api}}}
    return None


@pytest.mark.parametrize("abstract_name, expected_api", MATH_OPS)
def test_math_operations_backend(abstract_name: str, expected_api: str) -> None:
  """Test generating StableHLO for math operations via backend.

  Args:
      abstract_name: The Abstract operation name.
      expected_api: The expected stablehlo string.
  """
  nodes: List[LogicalNode] = [
    LogicalNode(id="in1", op_type="Input"),
    LogicalNode(id="op1", op_type=abstract_name, attributes={"test_attr": 42}),
    LogicalNode(id="out1", op_type="Output"),
  ]
  edges: List[LogicalEdge] = [
    LogicalEdge(source="in1", target="op1"),
    LogicalEdge(source="op1", target="out1"),
  ]
  graph: LogicalGraph = LogicalGraph(nodes={n.id: n for n in nodes}, edges=edges)
  backend: StableHloBackend = StableHloBackend(semantics=MathSemanticsMock())  # type: ignore
  code: str = backend.compile(graph)

  assert expected_api in code
  assert "test_attr = 42" in code


def test_stablehlo_emitter_math_ops() -> None:
  """Docstring."""
  semantics: SemanticsManager = SemanticsManager()
  # Assuming the semantics manager is loaded, but we can mock or inject
  # Actually, we can inject a mock directly.
  emitter: StableHloEmitter = StableHloEmitter(semantics)
  emitter._lookup_stablehlo_op = lambda x: x.replace("torch.", "stablehlo.") if x.startswith("torch.") else None  # type: ignore

  code: str = """
import torch
def test_math(x, y):
    a = torch.abs(x)
    b = torch.add(a, y)
    return b
    """
  tree: cst.Module = cst.parse_module(code)
  mlir_tree: Any = emitter.convert(tree)

  from ml_switcheroo.core.compiler.backends.mlir_printer import MlirPrinter

  out: str = MlirPrinter().emit(mlir_tree)

  assert "stablehlo.abs" in out
  assert "stablehlo.add" in out
