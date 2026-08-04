"""Test suite for the Lifter module."""

from typing import List
from ml_switcheroo.core.compiler.frontends.sass.lifter import SassLifter
from ml_switcheroo.core.compiler.frontends.sass.cst import SassComment, SassInstruction, SassRegister, SassNode


def test_lift_simple_chain() -> None:
  """Lifts simple chain."""
  nodes: List[SassNode] = [
    SassComment(text="Input x -> R0"),
    SassInstruction(opcode="MOV", operands=[SassRegister(name="R1"), SassRegister(name="RZ")]),
    SassComment(text="BEGIN Conv2d (conv1)"),
    SassInstruction(opcode="FADD", operands=[SassRegister(name="R1"), SassRegister(name="R1"), SassRegister(name="R0")]),
    SassComment(text="END Conv2d (conv1)"),
    SassComment(text="Return: R1"),
  ]
  lifter = SassLifter()
  graph = lifter.lift(nodes)
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
  """Lifts complex snippet."""
  nodes: List[SassNode] = [
    SassComment(text="Input x -> R0"),
    SassComment(text="BEGIN Conv2d (conv)"),
    SassComment(text="END Conv2d (conv)"),
    SassComment(text="Unmapped Op: torch.flatten (func_flatten)"),
    SassComment(text="BEGIN Linear (fc)"),
    SassComment(text="END Linear (fc)"),
    SassComment(text="Return: R7"),
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
  """Lifts duplicate markers ignored."""
  nodes: List[SassNode] = [
    SassComment(text="BEGIN Layer (l1)"),
    SassComment(text="END Layer (l1)"),
    SassComment(text="BEGIN Layer (l1)"),
    SassComment(text="END Layer (l1)"),
  ]
  lifter = SassLifter()
  graph = lifter.lift(nodes)
  assert len(graph.nodes) == 1
  assert graph.nodes[0].id == "l1"


def test_lift_no_comments() -> None:
  """Lifts no comments."""
  nodes: List[SassNode] = [
    SassInstruction(opcode="FADD", operands=[SassRegister(name="R0"), SassRegister(name="R1")]),
    SassComment(text="Just a normal comment"),
  ]
  lifter = SassLifter()
  graph = lifter.lift(nodes)
  assert len(graph.nodes) == 1
  assert graph.nodes[0].kind == "asm.FADD"
