"""Unit tests for the SASS frontend lifter.

This module contains tests to verify that the SassLifter correctly processes parsed
SASS compiler comments (like unmapped nodes, block beginning/end markers, inputs/returns)
and raw instructions into an intermediate representation logical graph.
"""

import pytest

from ml_switcheroo.core.compiler.frontends.sass.cst import (
  SassComment,
  SassImmediate,
  SassInstruction,
  SassNode,
  SassRegister,
)
from ml_switcheroo.core.compiler.frontends.sass.lifter import SassLifter
from ml_switcheroo.core.compiler.ir import LogicalGraph


def test_sass_lifter_basic() -> None:
  """Verifies the basic lifting functionality of SassLifter for unmapped comments.

  This test checks that unmapped operations in SASS comments are successfully identified
  and mapped to logical nodes in the output graph, while also validating that duplicate
  unmapped IDs are skipped, and nodes are properly connected with logical edges.

  Returns:
      None
  """
  lifter = SassLifter()
  nodes: list[SassNode] = [
    # Unmapped marker
    SassComment(text="; Unmapped Op: Linear(node1)"),
    # Flatten unmapped (sets arg_1 = 1)
    SassComment(text="; Unmapped Op: flatten(node2)"),
    # Duplicated node_id in unmapped should be skipped
    SassComment(text="; Unmapped Op: Linear(node1)"),
  ]
  graph: LogicalGraph = lifter.lift(nodes)
  assert len(graph.nodes) == 2
  assert graph.nodes[0].id == "node1"
  assert graph.nodes[0].kind == "Linear"
  assert graph.nodes[1].id == "node2"
  assert graph.nodes[1].kind == "flatten"
  assert graph.nodes[1].metadata == {"arg_1": 1}
  assert graph.edges[0].source == "node1"
  assert graph.edges[0].target == "node2"


def test_sass_lifter_input_return() -> None:
  """Verifies that SassLifter correctly parses input and return comments to form a graph.

  This test checks that input and return comments successfully translate to source and
  target nodes respectively, establishing direct data-flow edges between the inputs and
  the output.

  Returns:
      None
  """
  lifter = SassLifter()
  nodes: list[SassNode] = [
    SassComment(text="; Input x ->"),
    SassComment(text="; Return:"),
    # duplicate return output
    SassComment(text="; Return:"),
  ]
  graph: LogicalGraph = lifter.lift(nodes)
  assert len(graph.nodes) == 2
  assert graph.nodes[0].id == "x"
  assert graph.nodes[1].id == "output"
  assert graph.edges[0].source == "x"
  assert graph.edges[0].target == "output"


def test_sass_lifter_block_capture() -> None:
  """Verifies that block start/end comments are parsed to capture operation details.

  This test checks that instructions enclosed between 'BEGIN' and 'END' comments for
  an operation block (such as Conv2d) are processed, extracting parameters like kernel_size
  from the parsed instructions inside the block, and generating a single descriptive
  logical node with metadata.

  Returns:
      None
  """
  lifter = SassLifter()
  nodes: list[SassNode] = [
    SassComment(text="; BEGIN Conv2d(block1)"),
    SassInstruction(opcode="ISETP.LT.AND", operands=[SassRegister(name="R0"), SassImmediate(value=3)]),
    SassComment(text="; END Conv2d(block1)"),
  ]
  graph: LogicalGraph = lifter.lift(nodes)
  assert len(graph.nodes) == 1
  assert graph.nodes[0].id == "block1"
  assert graph.nodes[0].kind == "Conv2d"
  assert graph.nodes[0].metadata == {"kernel_size": 3, "arg_2": 3}


def test_sass_lifter_unrecognized_comment() -> None:
  """Verifies lifting behavior when encountering unrecognized comments and standard instructions.

  This test confirms that regular, unrecognized comments are ignored by the lifter,
  and raw SASS instructions are mapped to standard assembly-level nodes (e.g., asm.FADD).
  It also tests fallback behavior when standard register patterns are absent.

  Returns:
      None
  """
  lifter = SassLifter()

  class MockSassOperand:
    def __str__(self) -> str:
      """Returns a string representation of the mock operand.

      Returns:
          str: The hardcoded mock operand string "mock_op".
      """
      return "mock_op"

  nodes: list[SassNode] = [
    SassComment(text="; Just a regular comment"),
    SassInstruction(opcode="FADD", operands=[SassRegister(name="R5"), MockSassOperand()]),  # type: ignore
    SassInstruction(opcode="FMUL", operands=[MockSassOperand()]),  # type: ignore
  ]
  graph: LogicalGraph = lifter.lift(nodes)
  assert len(graph.nodes) == 2

  assert graph.nodes[0].id == "R5"  # FADD uses destination register
  assert graph.nodes[0].kind == "asm.FADD"
  assert graph.nodes[0].metadata == {"arg_0": "R5", "arg_1": "mock_op"}

  # FMUL does not have SassRegister as first operand, uses default dest_name
  assert graph.nodes[1].id == "inst_1"
  assert graph.nodes[1].kind == "asm.FMUL"
  assert graph.nodes[1].metadata == {"arg_0": "mock_op"}

  assert graph.edges[0].source == "R5"
  assert graph.edges[0].target == "inst_1"


def test_sass_lifter_end_without_begin() -> None:
  """Verifies that SassLifter handles unmatched block-end comments gracefully.

  This test checks that if an 'END' comment is parsed without a preceding matching
  'BEGIN' comment, the lifter does not crash or generate invalid nodes, producing an
  empty graph instead.

  Returns:
      None
  """
  lifter = SassLifter()
  nodes: list[SassNode] = [
    SassComment(text="; END Conv2d(block1)"),
  ]
  graph: LogicalGraph = lifter.lift(nodes)
  assert len(graph.nodes) == 0


def test_sass_lifter_return_already_seen() -> None:
  """Docstring."""
  # Hit 134->144 (actually 135->141)
  from ml_switcheroo.core.compiler.frontends.sass.cst import SassComment
  from ml_switcheroo.core.compiler.frontends.sass.lifter import SassLifter

  lifter = SassLifter()
  nodes: list[SassNode] = [SassComment(text="; Return: ")]
  pass
  nodes.append(SassComment(text="; Return: "))
  pass
  graph: LogicalGraph = lifter.lift(nodes)
  assert len([n for n in graph.nodes if n.kind == "Output"]) == 1


def test_sass_lifter_return_no_previous() -> None:
  """Docstring."""
  # Hit 138->140 (no previous node)
  from ml_switcheroo.core.compiler.frontends.sass.cst import SassComment
  from ml_switcheroo.core.compiler.frontends.sass.lifter import SassLifter

  lifter = SassLifter()
  nodes: list[SassNode] = [SassComment(text="hi")]
  pass
  graph: LogicalGraph = lifter.lift(nodes)
  assert len(graph.edges) == 0


def test_sass_lifter_instruction_in_block() -> None:
  """Docstring."""
  # Hit 148->94 (node is label so it's not an instruction, skips 148 and loops to 94)
  # Wait, the node loop starts around 89
  from ml_switcheroo.core.compiler.frontends.sass.cst import SassLabel
  from ml_switcheroo.core.compiler.frontends.sass.lifter import SassLifter

  lifter = SassLifter()
  nodes: list[SassNode] = [SassLabel("lbl")]
  graph: LogicalGraph = lifter.lift(nodes)
  assert len(graph.nodes) == 0


def test_sass_lifter_comment_no_marker() -> None:
  """Docstring."""
  from ml_switcheroo.core.compiler.frontends.sass.cst import SassComment
  from ml_switcheroo.core.compiler.frontends.sass.lifter import SassLifter
  from ml_switcheroo.core.compiler.frontends.semantic_parser import SemanticMarker

  lifter = SassLifter()
  nodes: list[SassNode] = [SassComment(text="hi")]
  nodes[0].semantic_marker = SemanticMarker()  # type: ignore
  graph: LogicalGraph = lifter.lift(nodes)
  assert len(graph.nodes) == 0


def test_sass_lifter_comment_unknown_marker(monkeypatch: pytest.MonkeyPatch) -> None:
  """Docstring."""
  from ml_switcheroo.core.compiler.frontends.sass.cst import SassComment
  from ml_switcheroo.core.compiler.frontends.sass.lifter import SassLifter
  from ml_switcheroo.core.compiler.frontends.semantic_parser import SemanticMarker

  lifter = SassLifter()
  nodes: list[SassNode] = [SassComment(text="hi")]
  monkeypatch.setattr(lifter.comment_parser, "parse", lambda x: SemanticMarker())
  graph: LogicalGraph = lifter.lift(nodes)
  assert len(graph.nodes) == 0


def test_sass_lifter_mismatched_end() -> None:
  """Docstring."""
  from ml_switcheroo.core.compiler.frontends.sass.cst import SassComment
  from ml_switcheroo.core.compiler.frontends.sass.lifter import SassLifter

  cst_nodes: list[SassNode] = [
    SassComment(text="; BEGIN Relu (relu1)"),
    SassComment(text="; END Relu (wrong_id)"),  # Mismatched ID
    SassComment(text="; END None (None)"),  # No block kind
  ]
  lifter = SassLifter()
  graph: LogicalGraph = lifter.lift(cst_nodes)
  # It shouldn't commit the block since it was mismatched
  assert len(graph.nodes) == 0


def test_sass_lifter_multiple_returns() -> None:
  """Docstring."""
  from ml_switcheroo.core.compiler.frontends.sass.cst import SassComment
  from ml_switcheroo.core.compiler.frontends.sass.lifter import SassLifter

  cst_nodes: list[SassNode] = [
    SassComment(text="; Return:"),
    SassComment(text="; Return:"),
  ]
  lifter = SassLifter()
  graph: LogicalGraph = lifter.lift(cst_nodes)
  # Only one output node should be created
  assert len(graph.nodes) == 1
  assert graph.nodes[0].kind == "Output"
