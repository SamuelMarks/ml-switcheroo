"""Test module."""

import pytest

from ml_switcheroo.analysis.cfg import BasicBlock, ControlFlowGraph


def test_basic_block() -> None:
  """Docstring."""
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
  """Docstring."""
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
  """Docstring."""
  cfg: ControlFlowGraph = ControlFlowGraph()
  cfg.get_or_create_block("block1")
  bb2: BasicBlock = cfg.get_or_create_block("block2")

  cfg.set_entry_block("block2")
  assert cfg.entry_block == bb2

  with pytest.raises(ValueError, match="Cannot set entry block: Block ID 'block3' not found in CFG."):
    cfg.set_entry_block("block3")


def test_cfg_traverse_dfs() -> None:
  """Docstring."""
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
  """Docstring."""
  cfg: ControlFlowGraph = ControlFlowGraph()
  with pytest.raises(ValueError, match="Cannot traverse CFG: No entry block defined."):
    cfg.traverse_dfs()
