from typing import List

from ml_switcheroo.compiler.frontends.sass.lifter import SassLifter
from ml_switcheroo.compiler.frontends.sass.nodes import Comment, Instruction, Register, SassNode


def test_lift_simple_chain() -> None:
  """
  Scenario: Input -> Conv2d -> Output.
  """
  nodes: List[SassNode] = [
    Comment("Input x -> R0"),
    Instruction("MOV", [Register("R1"), Register("RZ")]),
    Comment("BEGIN Conv2d (conv1)"),
    Instruction("FADD", [Register("R1"), Register("R1"), Register("R0")]),
    Comment("END Conv2d (conv1)"),
    Comment("Return: R1"),
  ]

  lifter = SassLifter()
  graph = lifter.lift(nodes)

  # Verify Nodes
  assert len(graph.nodes) == 4
  node_ids = [n.id for n in graph.nodes]

  assert "x" in node_ids
  assert "conv1" in node_ids
  assert "output" in node_ids
  assert "R1" in node_ids

  kinds = [n.kind for n in graph.nodes]
  assert "asm.MOV" in kinds
  assert "Conv2d" in kinds

  assert len(graph.edges) == 3


def test_lift_complex_snippet() -> None:
  """
  Scenario: Input -> Conv2d -> Flatten -> Linear -> Return.
  """
  nodes: List[SassNode] = [
    Comment("Input x -> R0"),
    Comment("BEGIN Conv2d (conv)"),
    Comment("END Conv2d (conv)"),
    Comment("Unmapped Op: torch.flatten (func_flatten)"),
    Comment("BEGIN Linear (fc)"),
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

  assert graph.edges[0].source == "x"
  assert graph.edges[0].target == "conv"


def test_lift_duplicate_markers_ignored() -> None:
  nodes: List[SassNode] = [
    Comment("BEGIN Layer (l1)"),
    Comment("END Layer (l1)"),
    Comment("BEGIN Layer (l1)"),
    Comment("END Layer (l1)"),
  ]

  lifter = SassLifter()
  graph = lifter.lift(nodes)

  assert len(graph.nodes) == 1
  assert graph.nodes[0].id == "l1"


def test_lift_no_comments() -> None:
  """
  Verify standard SASS without markers yields instructions nodes.
  """
  nodes: List[SassNode] = [
    Instruction("FADD", [Register("R0"), Register("R1")]),
    Comment("Just a normal comment"),
  ]

  lifter = SassLifter()
  graph = lifter.lift(nodes)

  assert len(graph.nodes) == 1
  assert graph.nodes[0].kind == "asm.FADD"
