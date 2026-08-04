"""Unit tests for the SASS Lifter frontend.

This module contains tests to verify that `SassLifter` correctly transforms low-level SASS
nodes (comments, instructions, registers, and immediates) into a unified `LogicalGraph`
with correct nodes, kind prefixes, metadata, and connectivity.
"""

from ml_switcheroo.core.compiler.frontends.sass.lifter import SassLifter
from ml_switcheroo.core.compiler.frontends.sass.cst import SassComment, SassInstruction, SassRegister, SassImmediate


def test_sass_lifter_unmapped():
  """Verify that SassLifter correctly processes unmapped custom operations from comments.

  This test checks that a SASS comment representing an unmapped operation (using the
  `// Unmapped Op:` format) is successfully parsed into a corresponding `LogicalNode`
  with the correct ID and operation kind in the returned `LogicalGraph`.

  Args:
    None

  Returns:
    None
  """
  lifter = SassLifter()
  nodes = [SassComment(text="// Unmapped Op: custom.op (custom_id)")]
  graph = lifter.lift(nodes)
  assert len(graph.nodes) == 1
  assert graph.nodes[0].id == "custom_id"
  assert graph.nodes[0].kind == "custom.op"


def test_sass_lifter_flatten():
  """Verify that SassLifter defaults the start_dim of a flattened op to 1 in PyTorch context.

  This test checks that when a comment contains the `torch.flatten` unmapped operation,
  the lifter correctly populates the metadata with "arg_1" set to 1, representing the default
  start_dim.

  Args:
    None

  Returns:
    None
  """
  lifter = SassLifter()
  nodes = [SassComment(text="// Unmapped Op: torch.flatten (flat_id)")]
  graph = lifter.lift(nodes)
  assert len(graph.nodes) == 1
  assert graph.nodes[0].metadata["arg_1"] == 1


def test_sass_lifter_instruction_only():
  """Verify that standalone ALU instructions are correctly parsed and converted to LogicalNodes.

  This test validates that standalone ALU instructions (e.g., IADD3) use the destination
  register as the node's unique identifier, prefix the opcode with "asm.", and capture
  all input registers correctly in the node's metadata.

  Args:
    None

  Returns:
    None
  """
  lifter = SassLifter()
  inst = SassInstruction(
    opcode="IADD3", operands=[SassRegister(name="R0"), SassRegister(name="R1"), SassRegister(name="R2")]
  )
  graph = lifter.lift([inst])
  assert len(graph.nodes) == 1
  assert graph.nodes[0].id == "R0"
  assert graph.nodes[0].kind == "asm.IADD3"
  assert graph.nodes[0].metadata["arg_0"] == "R0"
  assert graph.nodes[0].metadata["arg_1"] == "R1"
  assert graph.nodes[0].metadata["arg_2"] == "R2"


def test_sass_lifter_non_alu():
  """Verify that non-ALU standalone instructions generate generic node IDs and capture operands.

  This test checks that a branch instruction (e.g., BRA) which has no destination register
  is assigned a counter-based node ID (like "inst_0"), has its opcode prefixed with "asm.",
  and stores the correct operands (such as SassImmediate values) in its metadata.

  Args:
    None

  Returns:
    None
  """
  lifter = SassLifter()
  inst = SassInstruction(opcode="BRA", operands=[SassImmediate(value=10)])
  graph = lifter.lift([inst])
  assert len(graph.nodes) == 1
  assert graph.nodes[0].id == "inst_0"
  assert graph.nodes[0].kind == "asm.BRA"
  assert graph.nodes[0].metadata["arg_0"] == "10"


def test_sass_lifter_invalid_marker():
  """Verify that SASS comments without valid semantic markers are ignored during lifting.

  This test ensures that arbitrary comments that do not contain any recognizable semantic
  markers (e.g. unmapped ops, returns, blocks) are ignored, and do not result in any nodes
  being added to the output graph.

  Args:
    None

  Returns:
    None
  """
  lifter = SassLifter()
  nodes = [SassComment(text="// Just a comment without marker")]
  graph = lifter.lift(nodes)
  assert len(graph.nodes) == 0


def test_sass_lifter_return_already_seen():
  """Verify that duplicate return statements are deduplicated in the generated graph.

  This test ensures that when multiple return comment markers are encountered in the input,
  only a single output node with the ID "output" is registered in the graph to avoid duplicates.

  Args:
    None

  Returns:
    None
  """
  lifter = SassLifter()
  nodes = [
    SassComment(text="// Return: output"),
    SassComment(text="// Return: output"),
  ]
  graph = lifter.lift(nodes)
  assert len(graph.nodes) == 1
  assert graph.nodes[0].id == "output"


def test_sass_lifter_duplicate_node():
  """Verify that multiple duplicate nodes (e.g., inputs) are deduplicated.

  This test checks that if the same semantic node is declared multiple times (e.g.,
  multiple identical inputs), only the first declaration is added to the graph, and
  subsequent ones are correctly ignored to maintain uniqueness.

  Args:
    None

  Returns:
    None
  """
  lifter = SassLifter()
  nodes = [
    SassComment(text="// Input x -> x"),
    SassComment(text="// Input x -> x"),
  ]
  graph = lifter.lift(nodes)
  assert len(graph.nodes) == 1
  assert graph.nodes[0].id == "x"
