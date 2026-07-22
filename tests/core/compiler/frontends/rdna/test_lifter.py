"""Test suite for the Lifter module."""

from ml_switcheroo.core.compiler.frontends.rdna.nodes import Comment, Instruction, LabelRef
from ml_switcheroo.core.compiler.frontends.rdna.lifter import RdnaLifter
from unittest.mock import patch


def test_lifter_empty():
  """Verifies the behavior of lifter empty."""
  lifter = RdnaLifter()
  graph = lifter.lift([])
  assert len(graph.nodes) == 0


def test_lifter_input():
  """Verifies the behavior of lifter input."""
  lifter = RdnaLifter()
  nodes = [Comment("; Input x ->")]
  graph = lifter.lift(nodes)
  assert len(graph.nodes) == 1
  assert graph.nodes[0].kind == "Input"
  assert graph.nodes[0].id == "x"


def test_lifter_block():
  """Verifies the behavior of lifter block."""
  lifter = RdnaLifter()
  nodes = [
    Comment("; BEGIN Linear (l1)"),
    Instruction("v_add", [LabelRef("v1"), LabelRef("v2")]),
    Comment("; END Linear (l1)"),
  ]
  with patch("ml_switcheroo.core.compiler.frontends.rdna.analysis.RdnaAnalyzer.analyze_block") as mock_analyze:
    mock_analyze.return_value = {"features": 10}
    graph = lifter.lift(nodes)
  assert len(graph.nodes) == 1
  assert graph.nodes[0].id == "l1"
  assert graph.nodes[0].kind == "Linear"
  assert graph.nodes[0].metadata == {"features": 10}


def test_lifter_block_mismatch():
  """Verifies the behavior of lifter block mismatch."""
  lifter = RdnaLifter()
  nodes = [Comment("; BEGIN Linear (l1)"), Comment("; END Linear (wrong)")]
  graph = lifter.lift(nodes)
  assert len(graph.nodes) == 0


def test_lifter_unmapped():
  """Verifies the behavior of lifter unmapped."""
  lifter = RdnaLifter()
  nodes = [Comment("; Unmapped Op: torch.flatten (f1)")]
  graph = lifter.lift(nodes)
  assert len(graph.nodes) == 1
  assert graph.nodes[0].kind == "torch.flatten"
  assert graph.nodes[0].metadata == {"arg_1": 1}


def test_lifter_unmapped_no_flatten():
  """Verifies the behavior of lifter unmapped no flatten."""
  lifter = RdnaLifter()
  nodes = [Comment("; Unmapped Op: other.op (f2)")]
  graph = lifter.lift(nodes)
  assert len(graph.nodes) == 1
  assert graph.nodes[0].kind == "other.op"
  assert graph.nodes[0].metadata == {}


def test_lifter_return():
  """Verifies the behavior of lifter return."""
  lifter = RdnaLifter()
  nodes = [Comment("; Input x ->"), Comment("; Return:")]
  graph = lifter.lift(nodes)
  assert len(graph.nodes) == 2
  assert graph.nodes[1].kind == "Output"
  assert len(graph.edges) == 1
  assert graph.edges[0].source == "x"
  assert graph.edges[0].target == "output"


def test_lifter_raw_instruction():
  """Verifies the behavior of lifter raw instruction."""
  lifter = RdnaLifter()
  nodes = [Instruction("v_add", [LabelRef("v1"), LabelRef("v2")])]
  graph = lifter.lift(nodes)
  assert len(graph.nodes) == 1
  assert graph.nodes[0].kind == "rdna.v_add"


def test_lifter_duplicate_node():
  """Verifies the behavior of lifter duplicate node."""
  lifter = RdnaLifter()
  nodes = [Comment("; Input x ->"), Comment("; Input x ->")]
  graph = lifter.lift(nodes)
  assert len(graph.nodes) == 1


def test_lifter_return_no_previous():
  """Verifies the behavior of lifter return no previous."""
  lifter = RdnaLifter()
  nodes = [Comment("; Return:")]
  graph = lifter.lift(nodes)
  assert len(graph.nodes) == 1
  assert graph.nodes[0].kind == "Output"
  assert len(graph.edges) == 0
