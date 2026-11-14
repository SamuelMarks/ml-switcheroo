"""Test suite for the StableHLO backend."""

import pytest
from pydantic import JsonValue

from ml_switcheroo.core.compiler.backends.stablehlo import StableHloBackend
from ml_switcheroo.core.compiler.ir import LogicalEdge, LogicalGraph, LogicalNode
from ml_switcheroo.semantics.manager import SemanticsManager


class DummySemantics(SemanticsManager):
  """Test double `DummySemantics` implementing `get_definition` for backend stablehlo tests."""

  def __init__(self) -> None:
    """Initialize empty semantics manager without disk hydration."""
    super().__init__(load_defaults=False)

  def get_definition(self, api_name: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Lookup mock definition by operation kind.

    Args:
        api_name: Operation name to resolve.

    Returns:
        Definition tuple if matched, otherwise None.
    """
    if api_name == "Add":
      return "Add", {"variants": {"stablehlo": {"api": "stablehlo.add"}}}
    return None


def test_stablehlo_backend_init() -> None:
  """Verify stablehlo backend init stores the provided semantics instance."""
  sem = SemanticsManager.empty()
  backend: StableHloBackend = StableHloBackend(semantics=sem)
  assert backend.semantics is sem


def test_stablehlo_backend_compile() -> None:
  """Verify `LogicalGraph` emits `stablehlo.constant` for stablehlo backend compile."""
  nodes: list[LogicalNode] = [
    LogicalNode(id="in1", op_type="Input"),
    LogicalNode(id="in2", op_type="Input"),
    LogicalNode(id="add1", op_type="Add"),
    LogicalNode(id="mul1", op_type="Mul"),  # fallback to custom_call
    LogicalNode(id="out1", op_type="Output"),
  ]
  edges: list[LogicalEdge] = [
    LogicalEdge(source="in1", target="add1"),
    LogicalEdge(source="in2", target="add1"),
    LogicalEdge(source="add1", target="mul1"),
    LogicalEdge(source="in2", target="mul1"),
    LogicalEdge(source="mul1", target="out1"),
  ]
  graph: LogicalGraph = LogicalGraph(nodes={n.id: n for n in nodes}, edges=edges)
  backend: StableHloBackend = StableHloBackend(semantics=DummySemantics())
  code: str = backend.compile(graph)

  assert "Graph -> StableHLO compilation output" in code
  assert "stablehlo.constant" in code
  assert "stablehlo.add" in code
  assert "stablehlo.custom_call" in code
  assert '"@mul"' in code
  assert "return" in code


def test_stablehlo_backend_compile_no_semantics() -> None:
  """Verify `LogicalGraph` handles stablehlo backend compile no semantics as expected."""
  nodes: list[LogicalNode] = [
    LogicalNode(id="add1", op_type="Add"),
  ]
  edges: list[LogicalEdge] = []
  graph: LogicalGraph = LogicalGraph(nodes={n.id: n for n in nodes}, edges=edges)
  backend: StableHloBackend = StableHloBackend(semantics=None)
  code: str = backend.compile(graph)

  assert "stablehlo.custom_call" in code
  assert '"@add"' in code


OPS: list[tuple[str, str]] = [
  # CF
  ("Case", "stablehlo.case"),
  ("If", "stablehlo.if"),
  ("Map", "stablehlo.map"),
  ("While", "stablehlo.while"),
  # Comm
  ("AfterAll", "stablehlo.after_all"),
  ("AllGather", "stablehlo.all_gather"),
  ("AllReduce", "stablehlo.all_reduce"),
  ("AllToAll", "stablehlo.all_to_all"),
  ("AsyncDone", "stablehlo.async_done"),
  ("AsyncStart", "stablehlo.async_start"),
  ("CollectiveBroadcast", "stablehlo.collective_broadcast"),
  ("CollectivePermute", "stablehlo.collective_permute"),
  ("Infeed", "stablehlo.infeed"),
  ("Outfeed", "stablehlo.outfeed"),
  ("PartitionId", "stablehlo.partition_id"),
  ("Recv", "stablehlo.recv"),
  ("Reduce", "stablehlo.reduce"),
  ("ReducePrecision", "stablehlo.reduce_precision"),
  ("ReduceScatter", "stablehlo.reduce_scatter"),
  ("ReduceWindow", "stablehlo.reduce_window"),
  ("ReplicaId", "stablehlo.replica_id"),
  ("Send", "stablehlo.send"),
  # RNG
  ("Rng", "stablehlo.rng"),
  ("RngBitGenerator", "stablehlo.rng_bit_generator"),
  # Misc
  ("Clamp", "stablehlo.clamp"),
  ("Compare", "stablehlo.compare"),
  ("Composite", "stablehlo.composite"),
  ("Constant", "stablehlo.constant"),
  ("CustomCall", "stablehlo.custom_call"),
  ("IsFinite", "stablehlo.is_finite"),
  ("OptimizationBarrier", "stablehlo.optimization_barrier"),
  ("Select", "stablehlo.select"),
  ("SelectAndScatter", "stablehlo.select_and_scatter"),
  ("UniformDequantize", "stablehlo.uniform_dequantize"),
  ("UniformQuantize", "stablehlo.uniform_quantize"),
]


class RestSemanticsMock(SemanticsManager):
  """Mock semantics."""

  def __init__(self) -> None:
    """Initialize empty semantics manager without disk hydration."""
    super().__init__(load_defaults=False)

  def get_definition(self, api_name: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Get mock definition for kind.

    Args:
        api_name: Kind string.

    Returns:
        Definition tuple if matched, otherwise None.
    """
    for abstract, stablehlo_api in OPS:
      if api_name == abstract:
        return abstract, {"variants": {"stablehlo": {"api": stablehlo_api}}}
    return None


@pytest.mark.parametrize(("abstract_name", "expected_api"), OPS)
def test_rest_operations_backend(abstract_name: str, expected_api: str) -> None:
  """Test generating StableHLO via backend.

  Args:
      abstract_name (str): Abstract name.
      expected_api (str): Expected API string.
  """
  nodes: list[LogicalNode] = [
    LogicalNode(id="in1", op_type="Input"),
    LogicalNode(id="op1", op_type=abstract_name),
    LogicalNode(id="out1", op_type="Output"),
  ]
  edges: list[LogicalEdge] = [
    LogicalEdge(source="in1", target="op1"),
    LogicalEdge(source="op1", target="out1"),
  ]
  graph: LogicalGraph = LogicalGraph(nodes={n.id: n for n in nodes}, edges=edges)
  backend: StableHloBackend = StableHloBackend(semantics=RestSemanticsMock())
  code: str = backend.compile(graph)
  assert expected_api in code


class DummyStrSemantics(SemanticsManager):
  """Dummy semantics."""

  def __init__(self) -> None:
    """Initialize empty semantics manager without disk hydration."""
    super().__init__(load_defaults=False)

  def get_definition(self, api_name: str) -> tuple[str, dict[str, JsonValue]]:
    """Get mock definition.

    Args:
        api_name: Kind string.

    Returns:
        Definition tuple for dummy operation.
    """
    return api_name, {"variants": {"stablehlo": {"api": "stablehlo.dummy"}}}


def test_stablehlo_backend_str_attribute() -> None:
  """Verify `LogicalGraph` handles stablehlo backend str attribute as expected."""
  nodes: list[LogicalNode] = [
    LogicalNode(id="op1", op_type="Dummy", attributes={"str_attr": "hello_world", "quoted_attr": '"quoted"'})
  ]
  graph: LogicalGraph = LogicalGraph(nodes={n.id: n for n in nodes}, edges=[])
  backend: StableHloBackend = StableHloBackend(semantics=DummyStrSemantics())
  code: str = backend.compile(graph)
  assert 'str_attr = "hello_world"' in code
  assert 'quoted_attr = "quoted"' in code
