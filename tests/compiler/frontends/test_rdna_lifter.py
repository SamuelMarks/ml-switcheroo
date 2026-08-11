"""Tests for the RDNA lifter frontend of the ml_switcheroo compiler.

This module contains unit tests that verify the extraction of computation graphs
from AMD RDNA assembly instructions and comments, including unmapped operators,
input/return markers, block captures, and unrecognized assembly/comments.
"""

from ml_switcheroo.core.compiler.frontends.rdna.lifter import RdnaLifter
from ml_switcheroo.core.compiler.frontends.rdna.cst import (
  RdnaComment,
  RdnaInstruction,
  RdnaImmediate,
  RdnaVGPR,
)


def test_rdna_lifter_basic():
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
  nodes = [
    # Unmapped marker
    RdnaComment(text="; Unmapped Op: Linear(node1)"),
    # Flatten unmapped (sets arg_1 = 1)
    RdnaComment(text="; Unmapped Op: flatten(node2)"),
    # Duplicated node_id in unmapped should be skipped
    RdnaComment(text="; Unmapped Op: Linear(node1)"),
  ]
  graph = lifter.lift(nodes)
  assert len(graph.nodes) == 2
  assert graph.nodes[0].id == "node1"
  assert graph.nodes[0].kind == "Linear"
  assert graph.nodes[1].id == "node2"
  assert graph.nodes[1].kind == "flatten"
  assert graph.nodes[1].metadata == {"arg_1": 1}
  assert graph.edges[0].source == "node1"
  assert graph.edges[0].target == "node2"


def test_rdna_lifter_input_return():
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
  nodes = [
    RdnaComment(text="; Input x ->"),
    RdnaComment(text="; Return:"),
    # duplicate return output
    RdnaComment(text="; Return:"),
  ]
  graph = lifter.lift(nodes)
  assert len(graph.nodes) == 2
  assert graph.nodes[0].id == "x"
  assert graph.nodes[1].id == "output"
  assert graph.edges[0].source == "x"
  assert graph.edges[0].target == "output"


def test_rdna_lifter_block_capture():
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
  nodes = [
    RdnaComment(text="; BEGIN Conv2d(block1)"),
    RdnaInstruction(opcode="s_cmp_lt_i32", operands=[RdnaVGPR(0), RdnaImmediate(value=3)]),
    RdnaComment(text="; END Conv2d(block1)"),
  ]
  graph = lifter.lift(nodes)
  assert len(graph.nodes) == 1
  assert graph.nodes[0].id == "block1"
  assert graph.nodes[0].kind == "Conv2d"
  assert graph.nodes[0].metadata == {"k": 3, "arg_2": 3}


def test_rdna_lifter_unrecognized_comment():
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
  nodes = [
    RdnaComment(text="; Just a regular comment"),
    RdnaInstruction(opcode="v_add_f32"),
    RdnaInstruction(opcode="v_mul_f32"),
  ]
  graph = lifter.lift(nodes)
  assert len(graph.nodes) == 2
  assert graph.nodes[0].id == "inst_0"
  assert graph.nodes[0].kind == "rdna.v_add_f32"
  assert graph.nodes[1].id == "inst_1"
  assert graph.nodes[1].kind == "rdna.v_mul_f32"
  assert graph.edges[0].source == "inst_0"
  assert graph.edges[0].target == "inst_1"


def test_rdna_lifter_end_without_begin():
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
  nodes = [
    RdnaComment(text="; END Conv2d(block1)"),
  ]
  graph = lifter.lift(nodes)
  assert len(graph.nodes) == 0


def test_rdna_analysis_no_loop_limits():
  # Hit analysis 40
  """Test rdna analysis no loop limits."""
  from ml_switcheroo.core.compiler.frontends.rdna.analysis import RdnaAnalyzer
  from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaInstruction

  inst = RdnaInstruction(opcode="v_add_f32", operands=[])
  res = RdnaAnalyzer.analyze_block("Linear", [inst])
  assert res == {}


def test_rdna_analysis_other_kind():
  # Hit 47->53
  """Test rdna analysis other kind."""
  from ml_switcheroo.core.compiler.frontends.rdna.analysis import RdnaAnalyzer
  from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaInstruction, RdnaImmediate

  inst = RdnaInstruction(opcode="s_cmp_lt_i32", operands=[RdnaImmediate("10")])
  res = RdnaAnalyzer.analyze_block("OtherKind", [inst])
  assert res == {}


def test_rdna_lifter_return_already_seen():
  """Test rdna lifter return already seen."""
  from ml_switcheroo.core.compiler.frontends.rdna.lifter import RdnaLifter
  from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaComment

  lifter = RdnaLifter()
  nodes = [
    RdnaComment(text="; Return:"),
    RdnaComment(text="; Return:"),
  ]
  graph = lifter.lift(nodes)
  assert len([n for n in graph.nodes if n.kind == "Output"]) == 1


def test_rdna_lifter_return_no_previous():
  """Test rdna lifter return no previous."""
  from ml_switcheroo.core.compiler.frontends.rdna.lifter import RdnaLifter
  from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaComment

  lifter = RdnaLifter()
  nodes = [
    RdnaComment(text="; Return:"),
  ]
  graph = lifter.lift(nodes)
  assert len(graph.edges) == 0


def test_rdna_lifter_instruction_no_block_or_marker():
  """Test rdna lifter instruction no block or marker."""
  from ml_switcheroo.core.compiler.frontends.rdna.lifter import RdnaLifter
  from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaInstruction

  lifter = RdnaLifter()
  nodes = [RdnaInstruction(opcode="v_add_f32", operands=[])]
  graph = lifter.lift(nodes)
  assert len(graph.nodes) == 1
  assert graph.nodes[0].kind == "rdna.v_add_f32"


def test_rdna_lifter_instruction_in_block():
  """Test rdna lifter instruction in block."""
  from ml_switcheroo.core.compiler.frontends.rdna.lifter import RdnaLifter
  from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaLabel

  lifter = RdnaLifter()
  nodes = [RdnaLabel("lbl")]
  graph = lifter.lift(nodes)
  assert len(graph.nodes) == 0


def test_rdna_lifter_comment_no_marker():
  """Test rdna lifter comment no marker."""
  from ml_switcheroo.core.compiler.frontends.rdna.lifter import RdnaLifter
  from ml_switcheroo.core.compiler.frontends.semantic_parser import SemanticMarker
  from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaComment

  lifter = RdnaLifter()
  nodes = [RdnaComment(text="hi")]
  nodes[0].semantic_marker = SemanticMarker()
  graph = lifter.lift(nodes)
  assert len(graph.nodes) == 0


def test_rdna_lifter_comment_unknown_marker(monkeypatch):
  """Test rdna lifter comment unknown marker."""
  from ml_switcheroo.core.compiler.frontends.rdna.lifter import RdnaLifter
  from ml_switcheroo.core.compiler.frontends.semantic_parser import SemanticMarker
  from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaComment

  lifter = RdnaLifter()
  nodes = [RdnaComment(text="hi")]
  monkeypatch.setattr(lifter.comment_parser, "parse", lambda x: SemanticMarker())
  graph = lifter.lift(nodes)
  assert len(graph.nodes) == 0
