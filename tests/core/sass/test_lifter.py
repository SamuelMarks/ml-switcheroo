"""
Tests for the SASS Lifter analysis logic.

Verifies that SASS comments are correctly parsed into a LogicalGraph structure representing
the original model architecture.
"""

from typing import List

from ml_switcheroo.core.sass.lifter import SassLifter
from ml_switcheroo.core.sass.nodes import Comment, Instruction, Register, SassNode


def test_lift_simple_chain() -> None:
  """
  Scenario: Input -> Conv2d -> Output.
  """
  nodes: List[SassNode] = [
    Comment("Input x -> R0"),
    # Interleaved instructions shouldn't confuse the lifter
    Instruction("MOV", [Register("R1"), Register("RZ")]),
    Comment("BEGIN Conv2d (conv1)"),
    Instruction("FADD", [Register("R1"), Register("R1"), Register("R0")]),
    Comment("END Conv2d (conv1)"),
    Comment("Return: R1"),
  ]

  lifter = SassLifter()
  graph = lifter.lift(nodes)

  # Verify Nodes
  assert len(graph.nodes) == 3
  node_ids = [n.id for n in graph.nodes]
  assert node_ids == ["x", "conv1", "output"]

  # Verify Kinds
  kinds = [n.kind for n in graph.nodes]
  assert kinds == ["Input", "Conv2d", "Output"]

  # Verify Edges
  assert len(graph.edges) == 2
  assert graph.edges[0].source == "x"
  assert graph.edges[0].target == "conv1"
  assert graph.edges[1].source == "conv1"
  assert graph.edges[1].target == "output"


def test_lift_complex_snippet() -> None:
  """
  Scenario: The specific example provided in the prompt.
  Input -> Conv2d -> Flatten (Unmapped) -> Linear -> Return.
  """
  nodes: List[SassNode] = [
    Comment("Input x -> R0"),
    Comment("BEGIN Conv2d (conv)"),
    # ... opcodes ...
    Comment("END Conv2d (conv)"),
    Comment("Unmapped Op: torch.flatten (func_flatten)"),
    Comment("BEGIN Linear (fc)"),
    # ... opcodes ...
    Comment("END Linear (fc)"),
    Comment("Return: R7"),
  ]

  lifter = SassLifter()
  graph = lifter.lift(nodes)

  assert len(graph.nodes) == 5
  ids = [n.id for n in graph.nodes]
  assert ids == ["x", "conv", "func_flatten", "fc", "output"]

  kinds = [n.kind for n in graph.nodes]
  assert kinds == ["Input", "Conv2d", "torch.flatten", "Linear", "Output"]

  # Check Connectivity
  # x -> conv
  assert graph.edges[0].source == "x"
  assert graph.edges[0].target == "conv"
  # conv -> func_flatten
  assert graph.edges[1].source == "conv"
  assert graph.edges[1].target == "func_flatten"
  # func_flatten -> fc
  assert graph.edges[2].source == "func_flatten"
  assert graph.edges[2].target == "fc"
  # fc -> output
  assert graph.edges[3].source == "fc"
  assert graph.edges[3].target == "output"


def test_lift_duplicate_markers_ignored() -> None:
  """
  Since END and BEGIN might both be present, ensure we don't create duplicate nodes
  if the regexes accidentally matched both (though regex specifics prevent this, logic handles it).
  Wait, the lifter parses BEGIN. It ignores END currently.
  If we have multiple BEGINs for same ID (unlikely invalid SASS), it should ignore dupes.

  Update: The lifter now commits on END. So we need END markers.
  """
  nodes: List[SassNode] = [
    Comment("BEGIN Layer (l1)"),
    Comment("END Layer (l1)"),
    Comment("BEGIN Layer (l1)"),  # Duplicate block start (e.g. unrolled loop artifact?)
    Comment("END Layer (l1)"),  # Duplicate commit attempt
  ]

  lifter = SassLifter()
  graph = lifter.lift(nodes)

  assert len(graph.nodes) == 1
  assert graph.nodes[0].id == "l1"


def test_lift_no_comments() -> None:
  """Verify standard SASS without switcheroo markers yields empty graph."""
  nodes: List[SassNode] = [
    Instruction("FADD", [Register("R0"), Register("R1")]),
    Comment("Just a normal comment"),
  ]

  lifter = SassLifter()
  graph = lifter.lift(nodes)

  assert len(graph.nodes) == 0
