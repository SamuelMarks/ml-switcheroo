"""Test module for the Control Flow Graph (CFG) analysis components.

This module contains unit tests verifying the correctness of the `BasicBlock` and
`ControlFlowGraph` structures used for static analysis of code paths. It ensures
that blocks can be created, linked correctly, and that algorithms like DFS
traversal execute as expected.
"""

import pytest

from ml_switcheroo.analysis.cfg import BasicBlock, ControlFlowGraph


def test_basic_block() -> None:
  """Test the creation and mutation of a BasicBlock structure.

  Verifies that instructions can be appended to a basic block, and that predecessor
  and successor relationships are established and remain idempotent upon duplication.
  """
  bb1: BasicBlock = BasicBlock("block1")
  bb2: BasicBlock = BasicBlock("block2")

  assert bb1.id == "block1"
  assert len(bb1.instructions) == 0

  bb1.add_instruction("inst1")
  bb1.add_instruction("inst2")
  assert len(bb1.instructions) == 2
  assert bb1.instructions == ["inst1", "inst2"]

  bb1.add_successor(bb2)
  assert bb2 in bb1.successors
  assert bb1 in bb2.predecessors

  # Test idempotency of adding the same successor
  bb1.add_successor(bb2)
  assert len(bb1.successors) == 1
  assert len(bb2.predecessors) == 1


def test_control_flow_graph() -> None:
  """Test the core management logic of the ControlFlowGraph.

  Verifies that a CFG automatically designates the first created block as the
  entry block and ensures that subsequent requests for the same block ID
  return the existing block instance (singleton per ID).
  """
  cfg: ControlFlowGraph = ControlFlowGraph()
  assert cfg.entry_block is None

  bb1: BasicBlock = cfg.get_or_create_block("block1")
  assert cfg.entry_block == bb1

  bb2: BasicBlock = cfg.get_or_create_block("block2")
  bb1.add_successor(bb2)

  assert bb1.id == "block1"
  assert bb2.id == "block2"

  # Test getting existing block
  bb1_again: BasicBlock = cfg.get_or_create_block("block1")
  assert bb1 is bb1_again


def test_cfg_set_entry_block() -> None:
  """Test the explicit re-assignment of the CFG entry block.

  Verifies that the entry block can be changed dynamically to an existing block,
  and that attempting to set an unknown block as the entry raises an appropriate ValueError.
  """
  cfg: ControlFlowGraph = ControlFlowGraph()
  cfg.get_or_create_block("block1")
  bb2: BasicBlock = cfg.get_or_create_block("block2")

  cfg.set_entry_block("block2")
  assert cfg.entry_block == bb2

  with pytest.raises(ValueError, match="Cannot set entry block: Block ID 'block3' not found in CFG."):
    cfg.set_entry_block("block3")


def test_cfg_traverse_dfs() -> None:
  """Test the Depth-First Search (DFS) traversal algorithm on the CFG.

  Verifies that a standard diamond CFG structure (A -> B, A -> C, B -> D, C -> D)
  traverses correctly. Also tests that starting the traversal from an intermediate
  block yields only the reachable subset.
  """
  cfg: ControlFlowGraph = ControlFlowGraph()
  bb1: BasicBlock = cfg.get_or_create_block("A")
  bb2: BasicBlock = cfg.get_or_create_block("B")
  bb3: BasicBlock = cfg.get_or_create_block("C")
  bb4: BasicBlock = cfg.get_or_create_block("D")

  bb1.add_successor(bb2)
  bb1.add_successor(bb3)
  bb2.add_successor(bb4)
  bb3.add_successor(bb4)

  # A -> B -> D
  # |-> C -> D

  traversal: list[BasicBlock] = cfg.traverse_dfs()
  assert len(traversal) == 4
  assert traversal[0] == bb1
  assert traversal[1] == bb2
  assert traversal[2] == bb4
  assert traversal[3] == bb3

  # Traverse from a specific block
  traversal_sub: list[BasicBlock] = cfg.traverse_dfs(start_block=bb2)
  assert len(traversal_sub) == 2
  assert traversal_sub[0] == bb2
  assert traversal_sub[1] == bb4


def test_cfg_traverse_dfs_no_entry() -> None:
  """Test DFS traversal failure when no entry block is defined.

  Verifies that calling DFS on an empty ControlFlowGraph safely raises a ValueError.
  """
  cfg: ControlFlowGraph = ControlFlowGraph()
  with pytest.raises(ValueError, match="Cannot traverse CFG: No entry block defined."):
    cfg.traverse_dfs()
