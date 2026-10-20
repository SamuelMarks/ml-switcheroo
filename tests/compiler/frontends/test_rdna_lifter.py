"""Test suite for rdna lifter."""

import pytest


from ml_switcheroo.core.compiler.frontends.rdna.cst import (
  RdnaComment,
  RdnaImmediate,
  RdnaInstruction,
  RdnaLabel,
  RdnaVGPR,
)
from ml_switcheroo.core.compiler.frontends.rdna.lifter import RdnaLifter


def test_rdna_lifter_basic() -> None:
  """Tests the basic lifting capability of RdnaLifter for unmapped operations."""
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
  node_list = list(graph.nodes.values())
  assert node_list[0].id == "node1"
  assert node_list[0].op_type == "Linear"
  assert node_list[1].id == "node2"
  assert node_list[1].op_type == "flatten"
  assert node_list[1].attributes == {"arg_1": 1}
  assert graph.edges[0].source == "node1"
  assert graph.edges[0].target == "node2"


def test_rdna_lifter_input_return() -> None:
  """Tests parsing of input and return marker comments in assembly."""
  lifter = RdnaLifter()
  nodes = [
    RdnaComment(text="; Input x ->"),
    RdnaComment(text="; Return:"),
    # duplicate return output
    RdnaComment(text="; Return:"),
  ]
  graph = lifter.lift(nodes)
  assert len(graph.nodes) == 2
  node_list = list(graph.nodes.values())
  assert node_list[0].id == "x"
  assert node_list[1].id == "output"
  assert graph.edges[0].source == "x"
  assert graph.edges[0].target == "output"


def test_rdna_lifter_block_capture() -> None:
  """Tests capturing of operations within a block begin/end comment pair."""
  lifter = RdnaLifter()
  nodes = [
    RdnaComment(text="; BEGIN Conv2d(block1)"),
    RdnaInstruction(opcode="s_cmp_lt_i32", operands=[RdnaVGPR(0), RdnaImmediate(value=3)]),
    RdnaComment(text="; END Conv2d(block1)"),
  ]
  graph = lifter.lift(nodes)
  assert len(graph.nodes) == 1
  node_list = list(graph.nodes.values())
  assert node_list[0].id == "block1"
  assert node_list[0].op_type == "Conv2d"
  assert node_list[0].attributes == {"k": 3, "arg_2": 3}


def test_rdna_lifter_unrecognized_comment() -> None:
  """Tests lifting of instructions in the presence of unrecognized/generic comments."""
  lifter = RdnaLifter()
  nodes = [
    RdnaComment(text="; Just a regular comment"),
    RdnaInstruction(opcode="v_add_f32"),
    RdnaInstruction(opcode="v_mul_f32"),
  ]
  graph = lifter.lift(nodes)
  assert len(graph.nodes) == 2
  node_list = list(graph.nodes.values())
  assert node_list[0].id == "inst_0"
  assert node_list[0].op_type == "rdna.v_add_f32"
  assert node_list[1].id == "inst_1"
  assert node_list[1].op_type == "rdna.v_mul_f32"
  assert graph.edges[0].source == "inst_0"
  assert graph.edges[0].target == "inst_1"


def test_rdna_lifter_end_without_begin() -> None:
  """Tests that a block END comment without a preceding BEGIN comment is ignored."""
  lifter = RdnaLifter()
  nodes = [
    RdnaComment(text="; END Conv2d(block1)"),
  ]
  graph = lifter.lift(nodes)
  assert len(graph.nodes) == 0


def test_rdna_analysis_no_loop_limits() -> None:
  """Verify rdna analysis no loop limits behavior."""
  from ml_switcheroo.core.compiler.frontends.rdna.analysis import RdnaAnalyzer
  from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaInstruction

  inst = RdnaInstruction(opcode="v_add_f32", operands=[])
  res = RdnaAnalyzer.analyze_block("Linear", [inst])
  assert res == {}


def test_rdna_analysis_other_kind() -> None:
  """Verify rdna analysis other kind behavior."""
  from ml_switcheroo.core.compiler.frontends.rdna.analysis import RdnaAnalyzer
  from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaImmediate, RdnaInstruction

  inst = RdnaInstruction(opcode="s_cmp_lt_i32", operands=[RdnaImmediate("10")])  # type: ignore[arg-type]
  res = RdnaAnalyzer.analyze_block("OtherKind", [inst])
  assert res == {}


def test_rdna_lifter_return_already_seen() -> None:
  """Verify rdna lifter return already seen behavior."""
  from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaComment
  from ml_switcheroo.core.compiler.frontends.rdna.lifter import RdnaLifter

  lifter = RdnaLifter()
  nodes = [
    RdnaComment(text="; Return:"),
    RdnaComment(text="; Return:"),
  ]
  graph = lifter.lift(nodes)
  assert len([n for n in graph.nodes.values() if n.op_type == "Output"]) == 1


def test_rdna_lifter_return_no_previous() -> None:
  """Verify rdna lifter return no previous behavior."""
  from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaComment
  from ml_switcheroo.core.compiler.frontends.rdna.lifter import RdnaLifter

  lifter = RdnaLifter()
  nodes = [
    RdnaComment(text="; Return:"),
  ]
  graph = lifter.lift(nodes)
  assert len(graph.edges) == 0


def test_rdna_lifter_instruction_no_block_or_marker() -> None:
  """Verify rdna lifter instruction no block or marker behavior."""
  from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaInstruction
  from ml_switcheroo.core.compiler.frontends.rdna.lifter import RdnaLifter

  lifter = RdnaLifter()
  nodes = [RdnaInstruction(opcode="v_add_f32", operands=[])]
  graph = lifter.lift(nodes)
  assert len(graph.nodes) == 1
  node_list = list(graph.nodes.values())
  assert node_list[0].op_type == "rdna.v_add_f32"


def test_rdna_lifter_instruction_in_block() -> None:
  """Verify rdna lifter instruction in block behavior."""
  from ml_switcheroo.core.compiler.frontends.rdna.lifter import RdnaLifter

  lifter = RdnaLifter()
  nodes = [RdnaLabel("lbl")]
  graph = lifter.lift(nodes)
  assert len(graph.nodes) == 0


def test_rdna_lifter_comment_no_marker() -> None:
  """Verify rdna lifter comment no marker behavior."""
  from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaComment
  from ml_switcheroo.core.compiler.frontends.rdna.lifter import RdnaLifter
  from ml_switcheroo.core.compiler.frontends.semantic_parser import SemanticMarker

  lifter = RdnaLifter()
  nodes = [RdnaComment(text="hi")]
  nodes[0].semantic_marker = SemanticMarker()
  graph = lifter.lift(nodes)
  assert len(graph.nodes) == 0


def test_rdna_lifter_comment_unknown_marker(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify rdna lifter comment unknown marker behavior.

  Args:
      monkeypatch: Test argument or fixture.
  """
  from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaComment
  from ml_switcheroo.core.compiler.frontends.rdna.lifter import RdnaLifter
  from ml_switcheroo.core.compiler.frontends.semantic_parser import SemanticMarker

  lifter = RdnaLifter()
  nodes = [RdnaComment(text="hi")]
  monkeypatch.setattr(lifter.comment_parser, "parse", lambda x: SemanticMarker())
  graph = lifter.lift(nodes)
  assert len(graph.nodes) == 0


def test_rdna_analyzer_linear_no_loop_limits() -> None:
  """Verify rdna analyzer linear no loop limits behavior."""
  from ml_switcheroo.core.compiler.frontends.rdna.analysis import RdnaAnalyzer

  analyzer = RdnaAnalyzer()
  metadata = analyzer.analyze_block("Linear", [])
  assert "in_features" not in metadata


def test_rdna_lifter_mismatched_end() -> None:
  """Verify rdna lifter mismatched end behavior."""
  from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaComment
  from ml_switcheroo.core.compiler.frontends.rdna.lifter import RdnaLifter

  cst_nodes = [
    RdnaComment(text="; BEGIN Relu (relu1)"),
    RdnaComment(text="; END Relu (wrong_id)"),  # Mismatched ID
    RdnaComment(text="; END None (None)"),  # No block kind
  ]
  lifter = RdnaLifter()
  graph = lifter.lift(cst_nodes)
  assert len(graph.nodes) == 0


def test_rdna_lifter_multiple_returns() -> None:
  """Verify rdna lifter multiple returns behavior."""
  from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaComment
  from ml_switcheroo.core.compiler.frontends.rdna.lifter import RdnaLifter

  cst_nodes = [
    RdnaComment(text="; Return:"),
    RdnaComment(text="; Return:"),
  ]
  lifter = RdnaLifter()
  graph = lifter.lift(cst_nodes)
  assert len(graph.nodes) == 1
  node_list = list(graph.nodes.values())
  assert node_list[0].op_type == "Output"
