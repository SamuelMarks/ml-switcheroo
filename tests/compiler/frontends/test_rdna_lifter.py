"""Tests for the RDNA lifter frontend of the ml_switcheroo compiler.

This module contains unit tests that verify the extraction of computation graphs
from AMD RDNA assembly instructions and comments, including unmapped operators,
input/return markers, block captures, and unrecognized assembly/comments.
"""

import typing

import pytest

from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaComment, RdnaImmediate, RdnaInstruction, RdnaNode, RdnaVGPR
from ml_switcheroo.core.compiler.frontends.rdna.lifter import RdnaLifter
from ml_switcheroo.core.compiler.ir import LogicalGraph


def test_rdna_lifter_basic() -> None:
  """Tests the basic lifting capability of RdnaLifter for unmapped operations.

  Verifies that basic unmapped operator comments (e.g., "; Unmapped Op: ...")
  are correctly parsed, duplicate nodes are properly skipped, metadata is
  properly extracted, and edges are correctly formed between the sequential nodes.

  Args:
      None

  Returns:
      None
  """
  lifter = RdnaLifter()
  nodes: list[RdnaNode] = [
    # Unmapped marker
    RdnaComment(text="; Unmapped Op: Linear(node1)"),
    # Flatten unmapped (sets arg_1 = 1)
    RdnaComment(text="; Unmapped Op: flatten(node2)"),
    # Duplicated node_id in unmapped should be skipped
    RdnaComment(text="; Unmapped Op: Linear(node1)"),
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


def test_rdna_lifter_input_return() -> None:
  """Tests parsing of input and return marker comments in assembly.

  Verifies that input comments ("; Input x ->") and return comments ("; Return:")
  are lifted into corresponding input and output graph nodes with directed edges
  connecting inputs to the return output, while handling redundant return comments.

  Args:
      None

  Returns:
      None
  """
  lifter = RdnaLifter()
  nodes: list[RdnaNode] = [
    RdnaComment(text="; Input x ->"),
    RdnaComment(text="; Return:"),
    # duplicate return output
    RdnaComment(text="; Return:"),
  ]
  graph: LogicalGraph = lifter.lift(nodes)
  assert len(graph.nodes) == 2
  assert graph.nodes[0].id == "x"
  assert graph.nodes[1].id == "output"
  assert graph.edges[0].source == "x"
  assert graph.edges[0].target == "output"


def test_rdna_lifter_block_capture() -> None:
  """Tests capturing of operations within a block begin/end comment pair.

  Verifies that RdnaLifter handles structured blocks marked by "; BEGIN ..."
  and "; END ..." comments, parsing metadata (like immediate operand constants)
  from inner instructions within the block, and generating a single composite
  high-level graph node.

  Args:
      None

  Returns:
      None
  """
  lifter = RdnaLifter()
  nodes: list[RdnaNode] = [
    RdnaComment(text="; BEGIN Conv2d(block1)"),
    RdnaInstruction(opcode="s_cmp_lt_i32", operands=[RdnaVGPR(0), RdnaImmediate(value=3)]),
    RdnaComment(text="; END Conv2d(block1)"),
  ]
  graph: LogicalGraph = lifter.lift(nodes)
  assert len(graph.nodes) == 1
  assert graph.nodes[0].id == "block1"
  assert graph.nodes[0].kind == "Conv2d"
  assert graph.nodes[0].metadata == {"k": 3, "arg_2": 3}


def test_rdna_lifter_unrecognized_comment() -> None:
  """Tests lifting of instructions in the presence of unrecognized/generic comments.

  Verifies that generic comments are ignored during lifting, and that actual
  unstructured assembly instructions (e.g., "v_add_f32", "v_mul_f32") are successfully
  lifted into corresponding instruction nodes with appropriate sequencing edges.

  Args:
      None

  Returns:
      None
  """
  lifter = RdnaLifter()
  nodes: list[RdnaNode] = [
    RdnaComment(text="; Just a regular comment"),
    RdnaInstruction(opcode="v_add_f32"),
    RdnaInstruction(opcode="v_mul_f32"),
  ]
  graph: LogicalGraph = lifter.lift(nodes)
  assert len(graph.nodes) == 2
  assert graph.nodes[0].id == "inst_0"
  assert graph.nodes[0].kind == "rdna.v_add_f32"
  assert graph.nodes[1].id == "inst_1"
  assert graph.nodes[1].kind == "rdna.v_mul_f32"
  assert graph.edges[0].source == "inst_0"
  assert graph.edges[0].target == "inst_1"


def test_rdna_lifter_end_without_begin() -> None:
  """Tests that a block END comment without a preceding BEGIN comment is ignored.

  Verifies the robustness of the lifter's state machine when encountering mismatched
  block markers, ensuring no erroneous nodes are created when an END block is found
  without an active BEGIN block.

  Args:
      None

  Returns:
      None
  """
  lifter = RdnaLifter()
  nodes: list[RdnaNode] = [
    RdnaComment(text="; END Conv2d(block1)"),
  ]
  graph: LogicalGraph = lifter.lift(nodes)
  assert len(graph.nodes) == 0


def test_rdna_analysis_no_loop_limits() -> None:
  """Docstring."""
  # Hit analysis 40
  from ml_switcheroo.core.compiler.frontends.rdna.analysis import RdnaAnalyzer
  from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaInstruction

  inst = RdnaInstruction(opcode="v_add_f32", operands=[])
  res: dict[str, typing.Any] = RdnaAnalyzer.analyze_block("Linear", [inst])
  assert res == {}


def test_rdna_analysis_other_kind() -> None:
  """Docstring."""
  # Hit 47->53
  from ml_switcheroo.core.compiler.frontends.rdna.analysis import RdnaAnalyzer
  from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaImmediate, RdnaInstruction

  inst = RdnaInstruction(opcode="s_cmp_lt_i32", operands=[RdnaImmediate("10")])  # type: ignore
  res: dict[str, typing.Any] = RdnaAnalyzer.analyze_block("OtherKind", [inst])
  assert res == {}


def test_rdna_lifter_return_already_seen() -> None:
  """Docstring."""
  from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaComment
  from ml_switcheroo.core.compiler.frontends.rdna.lifter import RdnaLifter

  lifter = RdnaLifter()
  nodes: list[RdnaNode] = [
    RdnaComment(text="; Return:"),
    RdnaComment(text="; Return:"),
  ]
  graph: LogicalGraph = lifter.lift(nodes)
  assert len([n for n in graph.nodes if n.kind == "Output"]) == 1


def test_rdna_lifter_return_no_previous() -> None:
  """Docstring."""
  from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaComment
  from ml_switcheroo.core.compiler.frontends.rdna.lifter import RdnaLifter

  lifter = RdnaLifter()
  nodes: list[RdnaNode] = [
    RdnaComment(text="; Return:"),
  ]
  graph: LogicalGraph = lifter.lift(nodes)
  assert len(graph.edges) == 0


def test_rdna_lifter_instruction_no_block_or_marker() -> None:
  """Docstring."""
  from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaInstruction
  from ml_switcheroo.core.compiler.frontends.rdna.lifter import RdnaLifter

  lifter = RdnaLifter()
  nodes: list[RdnaNode] = [RdnaInstruction(opcode="v_add_f32", operands=[])]
  graph: LogicalGraph = lifter.lift(nodes)
  assert len(graph.nodes) == 1
  assert graph.nodes[0].kind == "rdna.v_add_f32"


def test_rdna_lifter_instruction_in_block() -> None:
  """Docstring."""
  from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaLabel
  from ml_switcheroo.core.compiler.frontends.rdna.lifter import RdnaLifter

  lifter = RdnaLifter()
  nodes: list[RdnaNode] = [RdnaLabel("lbl")]
  graph: LogicalGraph = lifter.lift(nodes)
  assert len(graph.nodes) == 0


def test_rdna_lifter_comment_no_marker() -> None:
  """Docstring."""
  from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaComment
  from ml_switcheroo.core.compiler.frontends.rdna.lifter import RdnaLifter
  from ml_switcheroo.core.compiler.frontends.semantic_parser import SemanticMarker

  lifter = RdnaLifter()
  nodes: list[RdnaNode] = [RdnaComment(text="hi")]
  nodes[0].semantic_marker = SemanticMarker()  # type: ignore
  graph: LogicalGraph = lifter.lift(nodes)
  assert len(graph.nodes) == 0


def test_rdna_lifter_comment_unknown_marker(monkeypatch: pytest.MonkeyPatch) -> None:
  """Docstring."""
  from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaComment
  from ml_switcheroo.core.compiler.frontends.rdna.lifter import RdnaLifter
  from ml_switcheroo.core.compiler.frontends.semantic_parser import SemanticMarker

  lifter = RdnaLifter()
  nodes: list[RdnaNode] = [RdnaComment(text="hi")]
  monkeypatch.setattr(lifter.comment_parser, "parse", lambda x: SemanticMarker())
  graph: LogicalGraph = lifter.lift(nodes)
  assert len(graph.nodes) == 0


def test_rdna_analyzer_linear_no_loop_limits() -> None:
  """Docstring."""
  from ml_switcheroo.core.compiler.frontends.rdna.analysis import RdnaAnalyzer

  analyzer = RdnaAnalyzer()
  metadata: dict[str, typing.Any] = analyzer.analyze_block("Linear", [])
  assert "in_features" not in metadata


def test_rdna_lifter_mismatched_end() -> None:
  """Docstring."""
  from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaComment
  from ml_switcheroo.core.compiler.frontends.rdna.lifter import RdnaLifter

  cst_nodes: list[RdnaNode] = [
    RdnaComment(text="; BEGIN Relu (relu1)"),
    RdnaComment(text="; END Relu (wrong_id)"),  # Mismatched ID
    RdnaComment(text="; END None (None)"),  # No block kind
  ]
  lifter = RdnaLifter()
  graph: LogicalGraph = lifter.lift(cst_nodes)
  # It shouldn't commit the block since it was mismatched
  assert len(graph.nodes) == 0


def test_rdna_lifter_multiple_returns() -> None:
  """Docstring."""
  from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaComment
  from ml_switcheroo.core.compiler.frontends.rdna.lifter import RdnaLifter

  cst_nodes: list[RdnaNode] = [
    RdnaComment(text="; Return:"),
    RdnaComment(text="; Return:"),
  ]
  lifter = RdnaLifter()
  graph: LogicalGraph = lifter.lift(cst_nodes)
  # Only one output node should be created
  assert len(graph.nodes) == 1
  assert graph.nodes[0].kind == "Output"
