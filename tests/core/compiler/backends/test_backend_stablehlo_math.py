"""Test suite for Math & Arithmetic operations in the StableHLO backend."""

import libcst as cst
import pytest
from pydantic import JsonValue

from ml_switcheroo.core.compiler.backends.stablehlo import StableHloBackend
from ml_switcheroo.core.compiler.ir import LogicalEdge, LogicalGraph, LogicalNode
from ml_switcheroo.core.mlir.stablehlo_emitter import StableHloEmitter
from ml_switcheroo.semantics.manager import SemanticsManager

MATH_OPS: list[tuple[str, str]] = [
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


class MathSemanticsMock(SemanticsManager):
  """Mock semantics manager for math ops."""

  def __init__(self) -> None:
    """Initialize empty semantics manager without disk hydration."""
    super().__init__(load_defaults=False)

  def get_definition(self, api_name: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Mock lookup.

    Args:
        api_name: Kind string.

    Returns:
        Definition tuple if matched, otherwise None.
    """
    for abstract, stablehlo_api in MATH_OPS:
      if api_name == abstract:
        return abstract, {"variants": {"stablehlo": {"api": stablehlo_api}}}
    return None


@pytest.mark.parametrize(("abstract_name", "expected_api"), MATH_OPS)
def test_math_operations_backend(abstract_name: str, expected_api: str) -> None:
  """Test generating StableHLO for math operations via backend.

  Args:
      abstract_name: The Abstract operation name.
      expected_api: The expected stablehlo string.
  """
  nodes: list[LogicalNode] = [
    LogicalNode(id="in1", op_type="Input"),
    LogicalNode(id="op1", op_type=abstract_name, attributes={"test_attr": 42}),
    LogicalNode(id="out1", op_type="Output"),
  ]
  edges: list[LogicalEdge] = [
    LogicalEdge(source="in1", target="op1"),
    LogicalEdge(source="op1", target="out1"),
  ]
  graph: LogicalGraph = LogicalGraph(nodes={n.id: n for n in nodes}, edges=edges)
  backend: StableHloBackend = StableHloBackend(semantics=MathSemanticsMock())
  code: str = backend.compile(graph)

  assert expected_api in code
  assert "test_attr = 42" in code


def test_stablehlo_emitter_math_ops(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `SemanticsManager` handles stablehlo emitter math operations as expected.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  semantics: SemanticsManager = SemanticsManager()
  # Assuming the semantics manager is loaded, but we can mock or inject
  # Actually, we can inject a mock directly.
  emitter: StableHloEmitter = StableHloEmitter(semantics)
  monkeypatch.setattr(
    emitter,
    "_lookup_stablehlo_op",
    lambda x: x.replace("torch.", "stablehlo.") if x.startswith("torch.") else None,
    raising=False,
  )

  code: str = """
import torch
def test_math(x, y):
    a = torch.abs(x)
    b = torch.add(a, y)
    return b
    """
  tree: cst.Module = cst.parse_module(code)
  mlir_tree = emitter.convert(tree)

  from ml_switcheroo.core.compiler.backends.mlir_printer import MlirPrinter

  out: str = MlirPrinter().emit(mlir_tree)

  assert "stablehlo.abs" in out
  assert "stablehlo.add" in out
