"""Unit tests for the NVIDIA_SASS Lifter frontend.

This module contains tests to verify that `NvidiaSassLifter` correctly transforms low-level NVIDIA_SASS
nodes (comments, instructions, registers, and immediates) into a unified `LogicalGraph`
with correct nodes, kind prefixes, metadata, and connectivity.
"""

from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import (
  NvidiaSassComment,
  NvidiaSassImmediate,
  NvidiaSassInstruction,
  NvidiaSassNode,
  NvidiaSassRegister,
)
from ml_switcheroo.core.compiler.frontends.nvidia_sass.lifter import NvidiaSassLifter
from ml_switcheroo.core.compiler.ir import LogicalGraph


def test_nvidia_sass_lifter_unmapped() -> None:
  """Verify that NvidiaSassLifter correctly processes unmapped custom operations from comments.

  This test checks that a NVIDIA_SASS comment representing an unmapped operation (using the
  `// Unmapped Op:` format) is successfully parsed into a corresponding `LogicalNode`
  with the correct ID and operation kind in the returned `LogicalGraph`.

  Args:
      None

  Returns:
      None
  """
  lifter = NvidiaSassLifter()
  nodes: list[NvidiaSassNode] = [NvidiaSassComment(text="// Unmapped Op: custom.op (custom_id)")]
  graph: LogicalGraph = lifter.lift(nodes)
  assert len(graph.nodes) == 1
  assert graph.nodes[0].id == "custom_id"
  assert graph.nodes[0].kind == "custom.op"


def test_nvidia_sass_lifter_flatten() -> None:
  """Verify that NvidiaSassLifter defaults the start_dim of a flattened op to 1 in PyTorch context.

  This test checks that when a comment contains the `torch.flatten` unmapped operation,
  the lifter correctly populates the metadata with "arg_1" set to 1, representing the default
  start_dim.

  Args:
      None

  Returns:
      None
  """
  lifter = NvidiaSassLifter()
  nodes: list[NvidiaSassNode] = [NvidiaSassComment(text="// Unmapped Op: torch.flatten (flat_id)")]
  graph: LogicalGraph = lifter.lift(nodes)
  assert len(graph.nodes) == 1
  assert graph.nodes[0].metadata["arg_1"] == 1


def test_nvidia_sass_lifter_instruction_only() -> None:
  """Verify that standalone ALU instructions are correctly parsed and converted to LogicalNodes.

  This test validates that standalone ALU instructions (e.g., IADD3) use the destination
  register as the node's unique identifier, prefix the opcode with "asm.", and capture
  all input registers correctly in the node's metadata.

  Args:
      None

  Returns:
      None
  """
  lifter = NvidiaSassLifter()
  inst = NvidiaSassInstruction(
    opcode="IADD3", operands=[NvidiaSassRegister(name="R0"), NvidiaSassRegister(name="R1"), NvidiaSassRegister(name="R2")]
  )
  graph: LogicalGraph = lifter.lift([inst])
  assert len(graph.nodes) == 1
  assert graph.nodes[0].id == "R0"
  assert graph.nodes[0].kind == "asm.IADD3"
  assert graph.nodes[0].metadata["arg_0"] == "R0"
  assert graph.nodes[0].metadata["arg_1"] == "R1"
  assert graph.nodes[0].metadata["arg_2"] == "R2"


def test_nvidia_sass_lifter_non_alu() -> None:
  """Verify that non-ALU standalone instructions generate generic node IDs and capture operands.

  This test checks that a branch instruction (e.g., BRA) which has no destination register
  is assigned a counter-based node ID (like "inst_0"), has its opcode prefixed with "asm.",
  and stores the correct operands (such as NvidiaSassImmediate values) in its metadata.

  Args:
      None

  Returns:
      None
  """
  lifter = NvidiaSassLifter()
  inst = NvidiaSassInstruction(opcode="BRA", operands=[NvidiaSassImmediate(value=10)])
  graph: LogicalGraph = lifter.lift([inst])
  assert len(graph.nodes) == 1
  assert graph.nodes[0].id == "inst_0"
  assert graph.nodes[0].kind == "asm.BRA"
  assert graph.nodes[0].metadata["arg_0"] == "10"


def test_nvidia_sass_lifter_invalid_marker() -> None:
  """Verify that NVIDIA_SASS comments without valid semantic markers are ignored during lifting.

  This test ensures that arbitrary comments that do not contain any recognizable semantic
  markers (e.g. unmapped ops, returns, blocks) are ignored, and do not result in any nodes
  being added to the output graph.

  Args:
      None

  Returns:
      None
  """
  lifter = NvidiaSassLifter()
  nodes: list[NvidiaSassNode] = [NvidiaSassComment(text="// Just a comment without marker")]
  graph: LogicalGraph = lifter.lift(nodes)
  assert len(graph.nodes) == 0


def test_nvidia_sass_lifter_return_already_seen() -> None:
  """Verify that duplicate return statements are deduplicated in the generated graph.

  This test ensures that when multiple return comment markers are encountered in the input,
  only a single output node with the ID "output" is registered in the graph to avoid duplicates.

  Args:
      None

  Returns:
      None
  """
  lifter = NvidiaSassLifter()
  nodes: list[NvidiaSassNode] = [
    NvidiaSassComment(text="// Return: output"),
    NvidiaSassComment(text="// Return: output"),
  ]
  graph: LogicalGraph = lifter.lift(nodes)
  assert len(graph.nodes) == 1
  assert graph.nodes[0].id == "output"


def test_nvidia_sass_lifter_duplicate_node() -> None:
  """Verify that multiple duplicate nodes (e.g., inputs) are deduplicated.

  This test checks that if the same semantic node is declared multiple times (e.g.,
  multiple identical inputs), only the first declaration is added to the graph, and
  subsequent ones are correctly ignored to maintain uniqueness.

  Args:
      None

  Returns:
      None
  """
  lifter = NvidiaSassLifter()
  nodes: list[NvidiaSassNode] = [
    NvidiaSassComment(text="// Input x -> x"),
    NvidiaSassComment(text="// Input x -> x"),
  ]
  graph: LogicalGraph = lifter.lift(nodes)
  assert len(graph.nodes) == 1
  assert graph.nodes[0].id == "x"
