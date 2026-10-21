"""Test suite for nvidia sass lifter."""

import pytest


from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import (
  NvidiaSassComment,
  NvidiaSassImmediate,
  NvidiaSassInstruction,
  NvidiaSassLabel,
  NvidiaSassNode,
  NvidiaSassRegister,
)
from ml_switcheroo.core.compiler.frontends.nvidia_sass.lifter import NvidiaSassLifter


def test_nvidia_sass_lifter_basic() -> None:
  """Verifies the basic lifting functionality of NvidiaSassLifter for unmapped comments."""
  lifter = NvidiaSassLifter()
  nodes = [
    # Unmapped marker
    NvidiaSassComment(text="; Unmapped Op: Linear(node1)"),
    # Flatten unmapped (sets arg_1 = 1)
    NvidiaSassComment(text="; Unmapped Op: flatten(node2)"),
    # Duplicated node_id in unmapped should be skipped
    NvidiaSassComment(text="; Unmapped Op: Linear(node1)"),
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


def test_nvidia_sass_lifter_input_return() -> None:
  """Verifies that NvidiaSassLifter correctly parses input and return comments to form a graph."""
  lifter = NvidiaSassLifter()
  nodes = [
    NvidiaSassComment(text="; Input x ->"),
    NvidiaSassComment(text="; Return:"),
    # duplicate return output
    NvidiaSassComment(text="; Return:"),
  ]
  graph = lifter.lift(nodes)
  assert len(graph.nodes) == 2
  node_list = list(graph.nodes.values())
  assert node_list[0].id == "x"
  assert node_list[1].id == "output"
  assert graph.edges[0].source == "x"
  assert graph.edges[0].target == "output"


def test_nvidia_sass_lifter_block_capture() -> None:
  """Verifies that block start/end comments are parsed to capture operation details."""
  lifter = NvidiaSassLifter()
  nodes = [
    NvidiaSassComment(text="; BEGIN Conv2d(block1)"),
    NvidiaSassInstruction(opcode="ISETP.LT.AND", operands=[NvidiaSassRegister(name="R0"), NvidiaSassImmediate(value=3)]),
    NvidiaSassComment(text="; END Conv2d(block1)"),
  ]
  graph = lifter.lift(nodes)
  assert len(graph.nodes) == 1
  node_list = list(graph.nodes.values())
  assert node_list[0].id == "block1"
  assert node_list[0].op_type == "Conv2d"
  assert node_list[0].attributes == {"kernel_size": 3, "arg_2": 3}


def test_nvidia_sass_lifter_unrecognized_comment() -> None:
  """Verifies lifting behavior when encountering unrecognized comments and standard instructions."""
  lifter = NvidiaSassLifter()

  class MockNvidiaSassOperand(NvidiaSassNode):
    """Test helper class MockNvidiaSassOperand."""

    def __str__(self) -> str:
      """Returns a string representation of the mock operand.

      Returns:
          str: The hardcoded mock operand string "mock_op".
      """
      return "mock_op"

  nodes = getattr(
    {
      "v": [
        NvidiaSassComment(text="; Just a regular comment"),
        getattr(
          {
            "v": getattr({"v": NvidiaSassInstruction}, "__getitem__")("v")(
              opcode=getattr({"v": "FADD"}, "__getitem__")("v"),
              operands=getattr(
                {
                  "v": [
                    getattr(
                      {
                        "v": getattr({"v": NvidiaSassRegister}, "__getitem__")("v")(
                          name=getattr({"v": "R5"}, "__getitem__")("v")
                        )
                      },
                      "__getitem__",
                    )("v"),
                    getattr({"v": getattr({"v": MockNvidiaSassOperand}, "__getitem__")("v")()}, "__getitem__")("v"),
                  ]
                },
                "__getitem__",
              )("v"),
            )
          },
          "__getitem__",
        )("v"),
        getattr(
          {
            "v": getattr({"v": NvidiaSassInstruction}, "__getitem__")("v")(
              opcode=getattr({"v": "FMUL"}, "__getitem__")("v"),
              operands=getattr(
                {"v": [getattr({"v": getattr({"v": MockNvidiaSassOperand}, "__getitem__")("v")()}, "__getitem__")("v")]},
                "__getitem__",
              )("v"),
            )
          },
          "__getitem__",
        )("v"),
      ]
    },
    "__getitem__",
  )("v")
  graph = lifter.lift(nodes)
  assert len(graph.nodes) == 2
  node_list = list(graph.nodes.values())

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
  lifter = NvidiaSassLifter()
  nodes = [
    NvidiaSassComment(text="; END Conv2d(block1)"),
  ]
  graph = lifter.lift(nodes)
  assert len(graph.nodes) == 0


def test_nvidia_sass_lifter_return_already_seen() -> None:
  """Verify nvidia sass lifter return already seen behavior."""
  from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassComment
  from ml_switcheroo.core.compiler.frontends.nvidia_sass.lifter import NvidiaSassLifter

  lifter = NvidiaSassLifter()
  nodes = [NvidiaSassComment(text="; Return: ")]
  nodes.append(NvidiaSassComment(text="; Return: "))
  graph = lifter.lift(nodes)
  assert len([n for n in graph.nodes.values() if n.op_type == "Output"]) == 1


def test_nvidia_sass_lifter_return_no_previous() -> None:
  """Verify nvidia sass lifter return no previous behavior."""
  from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassComment
  from ml_switcheroo.core.compiler.frontends.nvidia_sass.lifter import NvidiaSassLifter

  lifter = NvidiaSassLifter()
  nodes = [NvidiaSassComment(text="hi")]
  graph = lifter.lift(nodes)
  assert len(graph.edges) == 0


def test_nvidia_sass_lifter_instruction_in_block() -> None:
  """Verify nvidia sass lifter instruction in block behavior."""
  from ml_switcheroo.core.compiler.frontends.nvidia_sass.lifter import NvidiaSassLifter

  lifter = NvidiaSassLifter()
  nodes = [NvidiaSassLabel("lbl")]
  graph = lifter.lift(nodes)
  assert len(graph.nodes) == 0


def test_nvidia_sass_lifter_comment_no_marker() -> None:
  """Verify nvidia sass lifter comment no marker behavior."""
  from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassComment
  from ml_switcheroo.core.compiler.frontends.nvidia_sass.lifter import NvidiaSassLifter
  from ml_switcheroo.core.compiler.frontends.semantic_parser import SemanticMarker

  lifter = NvidiaSassLifter()
  nodes = [NvidiaSassComment(text="hi")]
  nodes[0].semantic_marker = SemanticMarker()
  graph = lifter.lift(nodes)
  assert len(graph.nodes) == 0


def test_nvidia_sass_lifter_comment_unknown_marker(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify nvidia sass lifter comment unknown marker behavior.

  Args:
      monkeypatch: Test argument or fixture.
  """
  from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassComment
  from ml_switcheroo.core.compiler.frontends.nvidia_sass.lifter import NvidiaSassLifter
  from ml_switcheroo.core.compiler.frontends.semantic_parser import SemanticMarker

  lifter = NvidiaSassLifter()
  nodes = [NvidiaSassComment(text="hi")]
  monkeypatch.setattr(lifter.comment_parser, "parse", lambda x: SemanticMarker())
  graph = lifter.lift(nodes)
  assert len(graph.nodes) == 0


def test_nvidia_sass_lifter_mismatched_end() -> None:
  """Verify nvidia sass lifter mismatched end behavior."""
  from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassComment
  from ml_switcheroo.core.compiler.frontends.nvidia_sass.lifter import NvidiaSassLifter

  cst_nodes = [
    NvidiaSassComment(text="; BEGIN Relu (relu1)"),
    NvidiaSassComment(text="; END Relu (wrong_id)"),  # Mismatched ID
    NvidiaSassComment(text="; END None (None)"),  # No block kind
  ]
  lifter = NvidiaSassLifter()
  graph = lifter.lift(cst_nodes)
  assert len(graph.nodes) == 0


def test_nvidia_sass_lifter_multiple_returns() -> None:
  """Verify nvidia sass lifter multiple returns behavior."""
  from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassComment
  from ml_switcheroo.core.compiler.frontends.nvidia_sass.lifter import NvidiaSassLifter

  cst_nodes = [
    NvidiaSassComment(text="; Return:"),
    NvidiaSassComment(text="; Return:"),
  ]
  lifter = NvidiaSassLifter()
  graph = lifter.lift(cst_nodes)
  assert len(graph.nodes) == 1
  node_list = list(graph.nodes.values())
  assert node_list[0].op_type == "Output"
