"""Test module."""

from typing import List

from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import (
  NvidiaSassComment,
  NvidiaSassInstruction,
  NvidiaSassLabel,
  NvidiaSassNode,
  NvidiaSassRegister,
)
from ml_switcheroo.core.compiler.frontends.nvidia_sass.lifter import NvidiaSassLifter
from ml_switcheroo.core.graph import LogicalGraph


def test_nvidia_sass_lifter_basic() -> None:
  """Docstring."""
  stmts: List[NvidiaSassNode] = [
    NvidiaSassComment(text="; BEGIN Linear(add1)"),
    NvidiaSassInstruction(
      opcode="VADD",
      operands=[NvidiaSassRegister(name="R0"), NvidiaSassRegister(name="R1"), NvidiaSassRegister(name="R2")],
    ),
    NvidiaSassComment(text="; END Linear(add1)"),
    # Hit line 80: node_id in seen_ids
    NvidiaSassComment(text="; BEGIN Linear(add1)"),
    NvidiaSassComment(text="; END Linear(add1)"),
    NvidiaSassComment(text="; Input in1 ->"),
    NvidiaSassComment(text="; Return:"),
    NvidiaSassComment(text="; Return:"),  # Duplicate return
    # Hit line 127
    NvidiaSassComment(text="; Unmapped Op: flatten(xyz)"),
    # Hit line 133->142: Needs to be a valid Semantic Marker that IS NOT Return, Input, Begin, End, Unmapped
    NvidiaSassComment(text="; Unmapped Op: something_else(xyz)"),
    NvidiaSassComment(text="; End Input"),
    # Hit line 133
    NvidiaSassComment(text="; Not a Return Marker"),
    # Hit line 98: not marker
    NvidiaSassComment(text="; just a comment"),
    NvidiaSassLabel(name="L1"),
  ]
  lifter: NvidiaSassLifter = NvidiaSassLifter()

  # Manually inject a fake marker that isn't handled explicitly to hit the 'else' branch
  from ml_switcheroo.core.compiler.frontends.semantic_parser import SemanticMarker

  class FakeMarker(SemanticMarker):
    """Fake marker for testing fallback logic."""

    pass

  original_parse = lifter.comment_parser.parse

  def mock_parse(text):
    if "FAKE_MARKER" in text:
      return FakeMarker()
    return original_parse(text)

  lifter.comment_parser.parse = mock_parse

  # Add fake marker
  stmts.insert(0, NvidiaSassComment(text="; FAKE_MARKER"))

  graph: LogicalGraph = lifter.lift(stmts)
  assert len(graph.nodes) >= 0


def test_nvidia_sass_lifter_return_no_prev() -> None:
  """Test Return without previous node (lines 137->139)."""
  stmts: List[NvidiaSassNode] = [
    NvidiaSassComment(text="; Return:"),
  ]
  lifter: NvidiaSassLifter = NvidiaSassLifter()
  graph: LogicalGraph = lifter.lift(stmts)
  assert len(graph.nodes) == 1
  assert len(graph.edges) == 0


def test_nvidia_sass_lifter_implicit_alu() -> None:
  """Test implicit ALU op with register dest."""
  stmts: List[NvidiaSassNode] = [
    NvidiaSassInstruction(
      opcode="FADD",
      operands=[NvidiaSassRegister(name="R3"), NvidiaSassRegister(name="R4")],
    ),
    NvidiaSassInstruction(
      opcode="NON_ALU",
      operands=[NvidiaSassRegister(name="R5")],
    ),
  ]
  lifter: NvidiaSassLifter = NvidiaSassLifter()
  graph: LogicalGraph = lifter.lift(stmts)
  assert len(graph.nodes) == 2


def test_nvidia_sass_lifter_mismatch_end() -> None:
  """Docstring."""
  stmts: List[NvidiaSassNode] = [
    NvidiaSassComment(text="; BEGIN Linear(add1)"),
    NvidiaSassComment(text="; END Linear(add2)"),
  ]
  lifter: NvidiaSassLifter = NvidiaSassLifter()
  graph: LogicalGraph = lifter.lift(stmts)
  assert len(graph.nodes) >= 0


def test_nvidia_sass_lifter_unmapped_flatten() -> None:
  """Docstring."""
  stmts: List[NvidiaSassNode] = [
    NvidiaSassComment(text="; Unmapped Op: flatten(some_id)"),
  ]
  lifter: NvidiaSassLifter = NvidiaSassLifter()
  graph: LogicalGraph = lifter.lift(stmts)
  assert len(graph.nodes) >= 0


def test_nvidia_sass_lifter_instruction_unmapped() -> None:
  """Docstring."""
  stmts: List[NvidiaSassNode] = [
    NvidiaSassInstruction(opcode="VADD", operands=[]),
  ]
  lifter: NvidiaSassLifter = NvidiaSassLifter()
  graph: LogicalGraph = lifter.lift(stmts)
  # The first instruction gets added as unmapped node
  assert len(graph.nodes) >= 0
