"""Test suite for the Lifter module."""

from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaComment, RdnaInstruction, RdnaLabelRef, RdnaNode
from ml_switcheroo.core.compiler.frontends.rdna.lifter import RdnaLifter
from ml_switcheroo.core.compiler.ir import LogicalGraph
from unittest.mock import patch


def test_lifter_empty() -> None:
  """Verifies the behavior of lifter empty."""
  lifter = RdnaLifter()
  graph: LogicalGraph = lifter.lift([])
  assert len(graph.nodes) == 0


def test_lifter_input() -> None:
  """Verifies the behavior of lifter input."""
  lifter = RdnaLifter()
  nodes: list[RdnaNode] = [RdnaComment(text="; Input x ->")]
  graph: LogicalGraph = lifter.lift(nodes)
  assert len(graph.nodes) == 1
  assert graph.nodes[0].kind == "Input"
  assert graph.nodes[0].id == "x"


def test_lifter_block() -> None:
  """Verifies the behavior of lifter block."""
  lifter = RdnaLifter()
  nodes: list[RdnaNode] = [
    RdnaComment(text="; BEGIN Linear (l1)"),
    RdnaInstruction(opcode="v_add", operands=[RdnaLabelRef(name="v1"), RdnaLabelRef(name="v2")]),
    RdnaComment(text="; END Linear (l1)"),
  ]
  with patch("ml_switcheroo.core.compiler.frontends.rdna.analysis.RdnaAnalyzer.analyze_block") as mock_analyze:
    mock_analyze.return_value = {"features": 10}
    graph: LogicalGraph = lifter.lift(nodes)
  assert len(graph.nodes) == 1
  assert graph.nodes[0].id == "l1"
  assert graph.nodes[0].kind == "Linear"
  assert graph.nodes[0].metadata == {"features": 10}


def test_lifter_block_mismatch() -> None:
  """Verifies the behavior of lifter block mismatch."""
  lifter = RdnaLifter()
  nodes: list[RdnaNode] = [RdnaComment(text="; BEGIN Linear (l1)"), RdnaComment(text="; END Linear (wrong)")]
  graph: LogicalGraph = lifter.lift(nodes)
  assert len(graph.nodes) == 0


def test_lifter_unmapped() -> None:
  """Verifies the behavior of lifter unmapped."""
  lifter = RdnaLifter()
  nodes: list[RdnaNode] = [RdnaComment(text="; Unmapped Op: torch.flatten (f1)")]
  graph: LogicalGraph = lifter.lift(nodes)
  assert len(graph.nodes) == 1
  assert graph.nodes[0].kind == "torch.flatten"
  assert graph.nodes[0].metadata == {"arg_1": 1}


def test_lifter_unmapped_no_flatten() -> None:
  """Verifies the behavior of lifter unmapped no flatten."""
  lifter = RdnaLifter()
  nodes: list[RdnaNode] = [RdnaComment(text="; Unmapped Op: other.op (f2)")]
  graph: LogicalGraph = lifter.lift(nodes)
  assert len(graph.nodes) == 1
  assert graph.nodes[0].kind == "other.op"
  assert graph.nodes[0].metadata == {}


def test_lifter_return() -> None:
  """Verifies the behavior of lifter return."""
  lifter = RdnaLifter()
  nodes: list[RdnaNode] = [RdnaComment(text="; Input x ->"), RdnaComment(text="; Return:")]
  graph: LogicalGraph = lifter.lift(nodes)
  assert len(graph.nodes) == 2
  assert graph.nodes[1].kind == "Output"
  assert len(graph.edges) == 1
  assert graph.edges[0].source == "x"
  assert graph.edges[0].target == "output"


def test_lifter_raw_instruction() -> None:
  """Verifies the behavior of lifter raw instruction."""
  lifter = RdnaLifter()
  nodes: list[RdnaNode] = [RdnaInstruction(opcode="v_add", operands=[RdnaLabelRef(name="v1"), RdnaLabelRef(name="v2")])]
  graph: LogicalGraph = lifter.lift(nodes)
  assert len(graph.nodes) == 1
  assert graph.nodes[0].kind == "rdna.v_add"


def test_lifter_duplicate_node() -> None:
  """Verifies the behavior of lifter duplicate node."""
  lifter = RdnaLifter()
  nodes: list[RdnaNode] = [RdnaComment(text="; Input x ->"), RdnaComment(text="; Input x ->")]
  graph: LogicalGraph = lifter.lift(nodes)
  assert len(graph.nodes) == 1


def test_lifter_return_no_previous() -> None:
  """Verifies the behavior of lifter return no previous."""
  lifter = RdnaLifter()
  nodes: list[RdnaNode] = [RdnaComment(text="; Return:")]
  graph: LogicalGraph = lifter.lift(nodes)
  assert len(graph.nodes) == 1
  assert graph.nodes[0].kind == "Output"
  assert len(graph.edges) == 0


def test_lifter_invalid_comment() -> None:
  """Verifies the behavior of lifter invalid comment."""
  lifter = RdnaLifter()
  nodes: list[RdnaNode] = [RdnaComment(text="; Just a normal comment")]
  graph: LogicalGraph = lifter.lift(nodes)
  assert len(graph.nodes) == 0
