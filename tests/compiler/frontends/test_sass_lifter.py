"""Unit tests for the SASS frontend lifter.

This module contains tests to verify that the SassLifter correctly processes parsed
SASS compiler comments (like unmapped nodes, block beginning/end markers, inputs/returns)
and raw instructions into an intermediate representation logical graph.
"""

from ml_switcheroo.core.compiler.frontends.sass.lifter import SassLifter
from ml_switcheroo.core.compiler.frontends.sass.cst import (
  SassComment,
  SassInstruction,
  SassImmediate,
  SassRegister,
)


def test_sass_lifter_basic():
  """Verifies the basic lifting functionality of SassLifter for unmapped comments.

  This test checks that unmapped operations in SASS comments are successfully identified
  and mapped to logical nodes in the output graph, while also validating that duplicate
  unmapped IDs are skipped, and nodes are properly connected with logical edges.

  Returns:
      None
  """
  lifter = SassLifter()
  nodes = [
    # Unmapped marker
    SassComment(text="; Unmapped Op: Linear(node1)"),
    # Flatten unmapped (sets arg_1 = 1)
    SassComment(text="; Unmapped Op: flatten(node2)"),
    # Duplicated node_id in unmapped should be skipped
    SassComment(text="; Unmapped Op: Linear(node1)"),
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


def test_sass_lifter_input_return():
  """Verifies that SassLifter correctly parses input and return comments to form a graph.

  This test checks that input and return comments successfully translate to source and
  target nodes respectively, establishing direct data-flow edges between the inputs and
  the output.

  Returns:
      None
  """
  lifter = SassLifter()
  nodes = [
    SassComment(text="; Input x ->"),
    SassComment(text="; Return:"),
    # duplicate return output
    SassComment(text="; Return:"),
  ]
  graph = lifter.lift(nodes)
  assert len(graph.nodes) == 2
  assert graph.nodes[0].id == "x"
  assert graph.nodes[1].id == "output"
  assert graph.edges[0].source == "x"
  assert graph.edges[0].target == "output"


def test_sass_lifter_block_capture():
  """Verifies that block start/end comments are parsed to capture operation details.

  This test checks that instructions enclosed between 'BEGIN' and 'END' comments for
  an operation block (such as Conv2d) are processed, extracting parameters like kernel_size
  from the parsed instructions inside the block, and generating a single descriptive
  logical node with metadata.

  Returns:
      None
  """
  lifter = SassLifter()
  nodes = [
    SassComment(text="; BEGIN Conv2d(block1)"),
    SassInstruction(opcode="ISETP.LT.AND", operands=[SassRegister(name="R0"), SassImmediate(value=3)]),
    SassComment(text="; END Conv2d(block1)"),
  ]
  graph = lifter.lift(nodes)
  assert len(graph.nodes) == 1
  assert graph.nodes[0].id == "block1"
  assert graph.nodes[0].kind == "Conv2d"
  assert graph.nodes[0].metadata == {"kernel_size": 3, "arg_2": 3}


def test_sass_lifter_unrecognized_comment():
  """Verifies lifting behavior when encountering unrecognized comments and standard instructions.

  This test confirms that regular, unrecognized comments are ignored by the lifter,
  and raw SASS instructions are mapped to standard assembly-level nodes (e.g., asm.FADD).
  It also tests fallback behavior when standard register patterns are absent.

  Returns:
      None
  """
  lifter = SassLifter()

  class MockSassOperand:
    """Mock implementation of a SASS instruction operand for testing fallback paths."""

    def __str__(self):
      """Returns a string representation of the mock operand.

      Returns:
          str: The hardcoded mock operand string "mock_op".
      """
      return "mock_op"

  nodes = [
    SassComment(text="; Just a regular comment"),
    SassInstruction(opcode="FADD", operands=[SassRegister(name="R5"), MockSassOperand()]),
    SassInstruction(opcode="FMUL", operands=[MockSassOperand()]),
  ]
  graph = lifter.lift(nodes)
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


def test_sass_lifter_end_without_begin():
  """Verifies that SassLifter handles unmatched block-end comments gracefully.

  This test checks that if an 'END' comment is parsed without a preceding matching
  'BEGIN' comment, the lifter does not crash or generate invalid nodes, producing an
  empty graph instead.

  Returns:
      None
  """
  lifter = SassLifter()
  nodes = [
    SassComment(text="; END Conv2d(block1)"),
  ]
  graph = lifter.lift(nodes)
  assert len(graph.nodes) == 0


def test_sass_lifter_return_already_seen():
  # Hit 134->144 (actually 135->141)
  """Test sass lifter return already seen."""
  from ml_switcheroo.core.compiler.frontends.sass.lifter import SassLifter
  from ml_switcheroo.core.compiler.frontends.sass.cst import SassComment

  lifter = SassLifter()
  nodes = [SassComment(text="; Return: ")]
  pass
  nodes.append(SassComment(text="; Return: "))
  pass
  graph = lifter.lift(nodes)
  assert len([n for n in graph.nodes if n.kind == "Output"]) == 1


def test_sass_lifter_return_no_previous():
  # Hit 138->140 (no previous node)
  """Test sass lifter return no previous."""
  from ml_switcheroo.core.compiler.frontends.sass.lifter import SassLifter
  from ml_switcheroo.core.compiler.frontends.sass.cst import SassComment

  lifter = SassLifter()
  nodes = [SassComment(text="hi")]
  pass
  graph = lifter.lift(nodes)
  assert len(graph.edges) == 0


def test_sass_lifter_instruction_in_block():
  # Hit 148->94 (node is label so it's not an instruction, skips 148 and loops to 94)
  # Wait, the node loop starts around 89
  """Test sass lifter instruction in block."""
  from ml_switcheroo.core.compiler.frontends.sass.lifter import SassLifter
  from ml_switcheroo.core.compiler.frontends.sass.cst import SassLabel

  lifter = SassLifter()
  nodes = [SassLabel("lbl")]
  graph = lifter.lift(nodes)
  assert len(graph.nodes) == 0


def test_sass_lifter_comment_no_marker():
  """Test sass lifter comment no marker."""
  from ml_switcheroo.core.compiler.frontends.sass.lifter import SassLifter
  from ml_switcheroo.core.compiler.frontends.semantic_parser import SemanticMarker
  from ml_switcheroo.core.compiler.frontends.sass.cst import SassComment

  lifter = SassLifter()
  nodes = [SassComment(text="hi")]
  nodes[0].semantic_marker = SemanticMarker()
  graph = lifter.lift(nodes)
  assert len(graph.nodes) == 0


def test_sass_lifter_comment_unknown_marker(monkeypatch):
  """Test sass lifter comment unknown marker."""
  from ml_switcheroo.core.compiler.frontends.sass.lifter import SassLifter
  from ml_switcheroo.core.compiler.frontends.semantic_parser import SemanticMarker
  from ml_switcheroo.core.compiler.frontends.sass.cst import SassComment

  lifter = SassLifter()
  nodes = [SassComment(text="hi")]
  monkeypatch.setattr(lifter.comment_parser, "parse", lambda x: SemanticMarker())
  graph = lifter.lift(nodes)
  assert len(graph.nodes) == 0
