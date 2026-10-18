"""Test suite for nvidia sass lifter."""

import typing


from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import (
  NvidiaSassComment,
  NvidiaSassImmediate,
  NvidiaSassInstruction,
  NvidiaSassLabel,
  NvidiaSassRegister,
)
from ml_switcheroo.core.compiler.frontends.nvidia_sass.lifter import NvidiaSassLifter


def test_nvidia_sass_lifter_basic() -> None:
  """Verifies the basic lifting functionality of NvidiaSassLifter for unmapped comments."""
  lifter: typing.Any = NvidiaSassLifter()
  nodes: typing.Any = [
    # Unmapped marker
    NvidiaSassComment(text="; Unmapped Op: Linear(node1)"),
    # Flatten unmapped (sets arg_1 = 1)
    NvidiaSassComment(text="; Unmapped Op: flatten(node2)"),
    # Duplicated node_id in unmapped should be skipped
    NvidiaSassComment(text="; Unmapped Op: Linear(node1)"),
  ]
  graph: typing.Any = lifter.lift(nodes)
  assert len(graph.nodes) == 2
  node_list: typing.Any = list(graph.nodes.values())
  assert node_list[0].id == "node1"
  assert node_list[0].op_type == "Linear"
  assert node_list[1].id == "node2"
  assert node_list[1].op_type == "flatten"
  assert node_list[1].attributes == {"arg_1": 1}
  assert graph.edges[0].source == "node1"
  assert graph.edges[0].target == "node2"


def test_nvidia_sass_lifter_input_return() -> None:
  """Verifies that NvidiaSassLifter correctly parses input and return comments to form a graph."""
  lifter: typing.Any = NvidiaSassLifter()
  nodes: typing.Any = [
    NvidiaSassComment(text="; Input x ->"),
    NvidiaSassComment(text="; Return:"),
    # duplicate return output
    NvidiaSassComment(text="; Return:"),
  ]
  graph: typing.Any = lifter.lift(nodes)
  assert len(graph.nodes) == 2
  node_list: typing.Any = list(graph.nodes.values())
  assert node_list[0].id == "x"
  assert node_list[1].id == "output"
  assert graph.edges[0].source == "x"
  assert graph.edges[0].target == "output"


def test_nvidia_sass_lifter_block_capture() -> None:
  """Verifies that block start/end comments are parsed to capture operation details."""
  lifter: typing.Any = NvidiaSassLifter()
  nodes: typing.Any = [
    NvidiaSassComment(text="; BEGIN Conv2d(block1)"),
    NvidiaSassInstruction(opcode="ISETP.LT.AND", operands=[NvidiaSassRegister(name="R0"), NvidiaSassImmediate(value=3)]),
    NvidiaSassComment(text="; END Conv2d(block1)"),
  ]
  graph: typing.Any = lifter.lift(nodes)
  assert len(graph.nodes) == 1
  node_list: typing.Any = list(graph.nodes.values())
  assert node_list[0].id == "block1"
  assert node_list[0].op_type == "Conv2d"
  assert node_list[0].attributes == {"kernel_size": 3, "arg_2": 3}


def test_nvidia_sass_lifter_unrecognized_comment() -> None:
  """Verifies lifting behavior when encountering unrecognized comments and standard instructions."""
  lifter: typing.Any = NvidiaSassLifter()

  class MockNvidiaSassOperand:
    """Test helper class MockNvidiaSassOperand."""

    def __str__(self) -> typing.Any:
      """Returns a string representation of the mock operand.

      Returns:
          str: The hardcoded mock operand string "mock_op".
      """
      return "mock_op"

  nodes: typing.Any = [
    NvidiaSassComment(text="; Just a regular comment"),
    NvidiaSassInstruction(opcode="FADD", operands=[NvidiaSassRegister(name="R5"), MockNvidiaSassOperand()]),
    NvidiaSassInstruction(opcode="FMUL", operands=[MockNvidiaSassOperand()]),
  ]
  graph: typing.Any = lifter.lift(nodes)
  assert len(graph.nodes) == 2
  node_list: typing.Any = list(graph.nodes.values())

  assert node_list[0].id == "R5"  # FADD uses destination register
  assert node_list[0].op_type == "asm.FADD"
  assert node_list[0].attributes == {"arg_0": "R5", "arg_1": "mock_op"}

  # FMUL does not have NvidiaSassRegister as first operand, uses default dest_name
  assert node_list[1].id == "inst_1"
  assert node_list[1].op_type == "asm.FMUL"
  assert node_list[1].attributes == {"arg_0": "mock_op"}

  assert graph.edges[0].source == "R5"
  assert graph.edges[0].target == "inst_1"


def test_nvidia_sass_lifter_end_without_begin() -> None:
  """Verifies that NvidiaSassLifter handles unmatched block-end comments gracefully."""
  lifter: typing.Any = NvidiaSassLifter()
  nodes: typing.Any = [
    NvidiaSassComment(text="; END Conv2d(block1)"),
  ]
  graph: typing.Any = lifter.lift(nodes)
  assert len(graph.nodes) == 0


def test_nvidia_sass_lifter_return_already_seen() -> None:
  """Verify nvidia sass lifter return already seen behavior."""
  from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassComment
  from ml_switcheroo.core.compiler.frontends.nvidia_sass.lifter import NvidiaSassLifter

  lifter: typing.Any = NvidiaSassLifter()
  nodes: typing.Any = [NvidiaSassComment(text="; Return: ")]
  nodes.append(NvidiaSassComment(text="; Return: "))
  graph: typing.Any = lifter.lift(nodes)
  assert len([n for n in graph.nodes.values() if n.op_type == "Output"]) == 1


def test_nvidia_sass_lifter_return_no_previous() -> None:
  """Verify nvidia sass lifter return no previous behavior."""
  from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassComment
  from ml_switcheroo.core.compiler.frontends.nvidia_sass.lifter import NvidiaSassLifter

  lifter: typing.Any = NvidiaSassLifter()
  nodes: typing.Any = [NvidiaSassComment(text="hi")]
  graph: typing.Any = lifter.lift(nodes)
  assert len(graph.edges) == 0


def test_nvidia_sass_lifter_instruction_in_block() -> None:
  """Verify nvidia sass lifter instruction in block behavior."""
  from ml_switcheroo.core.compiler.frontends.nvidia_sass.lifter import NvidiaSassLifter

  lifter: typing.Any = NvidiaSassLifter()
  nodes: typing.Any = [NvidiaSassLabel("lbl")]
  graph: typing.Any = lifter.lift(nodes)
  assert len(graph.nodes) == 0


def test_nvidia_sass_lifter_comment_no_marker() -> None:
  """Verify nvidia sass lifter comment no marker behavior."""
  from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassComment
  from ml_switcheroo.core.compiler.frontends.nvidia_sass.lifter import NvidiaSassLifter
  from ml_switcheroo.core.compiler.frontends.semantic_parser import SemanticMarker

  lifter: typing.Any = NvidiaSassLifter()
  nodes: typing.Any = [NvidiaSassComment(text="hi")]
  nodes[0].semantic_marker = SemanticMarker()
  graph: typing.Any = lifter.lift(nodes)
  assert len(graph.nodes) == 0


def test_nvidia_sass_lifter_comment_unknown_marker(monkeypatch: typing.Any) -> None:
  """Verify nvidia sass lifter comment unknown marker behavior.

  Args:
      monkeypatch: Test argument or fixture.
  """
  from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassComment
  from ml_switcheroo.core.compiler.frontends.nvidia_sass.lifter import NvidiaSassLifter
  from ml_switcheroo.core.compiler.frontends.semantic_parser import SemanticMarker

  lifter: typing.Any = NvidiaSassLifter()
  nodes: typing.Any = [NvidiaSassComment(text="hi")]
  monkeypatch.setattr(lifter.comment_parser, "parse", lambda x: SemanticMarker())
  graph: typing.Any = lifter.lift(nodes)
  assert len(graph.nodes) == 0


def test_nvidia_sass_lifter_mismatched_end() -> None:
  """Verify nvidia sass lifter mismatched end behavior."""
  from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassComment
  from ml_switcheroo.core.compiler.frontends.nvidia_sass.lifter import NvidiaSassLifter

  cst_nodes: typing.Any = [
    NvidiaSassComment(text="; BEGIN Relu (relu1)"),
    NvidiaSassComment(text="; END Relu (wrong_id)"),  # Mismatched ID
    NvidiaSassComment(text="; END None (None)"),  # No block kind
  ]
  lifter: typing.Any = NvidiaSassLifter()
  graph: typing.Any = lifter.lift(cst_nodes)
  assert len(graph.nodes) == 0


def test_nvidia_sass_lifter_multiple_returns() -> None:
  """Verify nvidia sass lifter multiple returns behavior."""
  from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassComment
  from ml_switcheroo.core.compiler.frontends.nvidia_sass.lifter import NvidiaSassLifter

  cst_nodes: typing.Any = [
    NvidiaSassComment(text="; Return:"),
    NvidiaSassComment(text="; Return:"),
  ]
  lifter: typing.Any = NvidiaSassLifter()
  graph: typing.Any = lifter.lift(cst_nodes)
  assert len(graph.nodes) == 1
  node_list: typing.Any = list(graph.nodes.values())
  assert node_list[0].op_type == "Output"
