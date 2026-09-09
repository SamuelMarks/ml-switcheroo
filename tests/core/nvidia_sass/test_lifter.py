"""Test suite for the Lifter module."""

from typing import List

from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import (
  NvidiaSassComment,
  NvidiaSassInstruction,
  NvidiaSassNode,
  NvidiaSassRegister,
)
from ml_switcheroo.core.compiler.frontends.nvidia_sass.lifter import NvidiaSassLifter


def test_lift_simple_chain() -> None:
  """Lifts simple chain."""
  nodes: List[NvidiaSassNode] = [
    NvidiaSassComment(text="Input x -> R0"),
    NvidiaSassInstruction(opcode="MOV", operands=[NvidiaSassRegister(name="R1"), NvidiaSassRegister(name="RZ")]),
    NvidiaSassComment(text="BEGIN Conv2d (conv1)"),
    NvidiaSassInstruction(
      opcode="FADD",
      operands=[NvidiaSassRegister(name="R1"), NvidiaSassRegister(name="R1"), NvidiaSassRegister(name="R0")],
    ),
    NvidiaSassComment(text="END Conv2d (conv1)"),
    NvidiaSassComment(text="Return: R1"),
  ]
  lifter = NvidiaSassLifter()
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
  nodes: List[NvidiaSassNode] = [
    NvidiaSassComment(text="Input x -> R0"),
    NvidiaSassComment(text="BEGIN Conv2d (conv)"),
    NvidiaSassComment(text="END Conv2d (conv)"),
    NvidiaSassComment(text="Unmapped Op: torch.flatten (func_flatten)"),
    NvidiaSassComment(text="BEGIN Linear (fc)"),
    NvidiaSassComment(text="END Linear (fc)"),
    NvidiaSassComment(text="Return: R7"),
  ]
  lifter = NvidiaSassLifter()
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
  nodes: List[NvidiaSassNode] = [
    NvidiaSassComment(text="BEGIN Layer (l1)"),
    NvidiaSassComment(text="END Layer (l1)"),
    NvidiaSassComment(text="BEGIN Layer (l1)"),
    NvidiaSassComment(text="END Layer (l1)"),
  ]
  lifter = NvidiaSassLifter()
  graph = lifter.lift(nodes)
  assert len(graph.nodes) == 1
  assert graph.nodes[0].id == "l1"


def test_lift_no_comments() -> None:
  """Lifts no comments."""
  nodes: List[NvidiaSassNode] = [
    NvidiaSassInstruction(opcode="FADD", operands=[NvidiaSassRegister(name="R0"), NvidiaSassRegister(name="R1")]),
    NvidiaSassComment(text="Just a normal comment"),
  ]
  lifter = NvidiaSassLifter()
  graph = lifter.lift(nodes)
  assert len(graph.nodes) == 1
  assert graph.nodes[0].kind == "asm.FADD"
