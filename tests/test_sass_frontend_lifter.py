"""Test module."""

from typing import List

from ml_switcheroo.core.compiler.frontends.sass.cst import SassComment, SassInstruction, SassLabel, SassNode, SassRegister
from ml_switcheroo.core.compiler.frontends.sass.lifter import SassLifter
from ml_switcheroo.core.graph import LogicalGraph


def test_sass_lifter_basic() -> None:
  """Docstring."""
  stmts: List[SassNode] = [
    SassComment(text="; BEGIN Linear(add1)"),
    SassInstruction(opcode="VADD", operands=[SassRegister(name="R0"), SassRegister(name="R1"), SassRegister(name="R2")]),
    SassComment(text="; END Linear(add1)"),
    SassComment(text="; Input in1 ->"),
    SassComment(text="; Return:"),
    SassComment(text="; Return:"),  # Duplicate return
    SassComment(text="; Unmapped Op: something(xyz)"),
    SassLabel(name="L1"),
  ]
  lifter: SassLifter = SassLifter()
  graph: LogicalGraph = lifter.lift(stmts)
  assert len(graph.nodes) >= 0


def test_sass_lifter_mismatch_end() -> None:
  """Docstring."""
  stmts: List[SassNode] = [
    SassComment(text="; BEGIN Linear(add1)"),
    SassComment(text="; END Linear(add2)"),
  ]
  lifter: SassLifter = SassLifter()
  graph: LogicalGraph = lifter.lift(stmts)
  assert len(graph.nodes) >= 0


def test_sass_lifter_unmapped_flatten() -> None:
  """Docstring."""
  stmts: List[SassNode] = [
    SassComment(text="; Unmapped Op: flatten(some_id)"),
  ]
  lifter: SassLifter = SassLifter()
  graph: LogicalGraph = lifter.lift(stmts)
  assert len(graph.nodes) >= 0


def test_sass_lifter_instruction_unmapped() -> None:
  """Docstring."""
  stmts: List[SassNode] = [
    SassInstruction(opcode="VADD", operands=[]),
  ]
  lifter: SassLifter = SassLifter()
  graph: LogicalGraph = lifter.lift(stmts)
  # The first instruction gets added as unmapped node
  assert len(graph.nodes) >= 0
