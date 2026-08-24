"""Control Flow Graph (CFG) analysis framework for assembly lifting.

This module provides the core data structures necessary to build and analyze
Control Flow Graphs from low-level assembly streams, replacing simple string
heuristics with topological graph analysis.
"""

from typing import List, Optional, Set, Dict, Any


class BasicBlock:
  """Represent a basic block in a Control Flow Graph.

  A basic block is a straight-line sequence of instructions with no branches
  in except to the entry, and no branches out except at the exit.

  Attributes:
      id: A unique identifier for the block (e.g., a label name or auto-incremented int).
      instructions: The sequence of instructions contained in this block.
      successors: A list of blocks that execution can flow to after this block.
      predecessors: A list of blocks that can flow into this block.
  """

  def __init__(self, block_id: str) -> None:
    """Initialize a new BasicBlock.

    Args:
        block_id: The unique identifier for this block.
    """
    self.id: str = block_id
    self.instructions: List[Any] = []
    self.successors: List["BasicBlock"] = []
    self.predecessors: List["BasicBlock"] = []

  def add_instruction(self, instruction: Any) -> None:
    """Append an instruction to the basic block.

    Args:
        instruction: The instruction object or string to append.
    """
    self.instructions.append(instruction)

  def add_successor(self, successor: "BasicBlock") -> None:
    """Add a successor block to this block's outgoing edges.

    Also updates the successor's predecessors list to maintain bidirectional linkage.

    Args:
        successor: The basic block that can follow this one.
    """
    if successor not in self.successors:
      self.successors.append(successor)
    if self not in successor.predecessors:
      successor.predecessors.append(self)


class ControlFlowGraph:
  """Represent a Control Flow Graph (CFG) composed of basic blocks.

  Attributes:
      blocks: A dictionary mapping block IDs to their BasicBlock instances.
      entry_block: The starting block of the graph, if defined.
  """

  def __init__(self) -> None:
    """Initialize an empty Control Flow Graph."""
    self.blocks: Dict[str, BasicBlock] = {}
    self.entry_block: Optional[BasicBlock] = None

  def get_or_create_block(self, block_id: str) -> BasicBlock:
    """Retrieve an existing block or creates a new one if it doesn't exist.

    Args:
        block_id: The unique identifier of the block.

    Returns:
        The existing or newly created BasicBlock.
    """
    if block_id not in self.blocks:
      self.blocks[block_id] = BasicBlock(block_id)
      # If this is the first block created, mark it as entry
      if self.entry_block is None:
        self.entry_block = self.blocks[block_id]
    return self.blocks[block_id]

  def set_entry_block(self, block_id: str) -> None:
    """Explicitly sets the entry block of the CFG.

    Args:
        block_id: The ID of the block to set as the entry point.

    Raises:
        ValueError: If the block_id does not exist in the graph.
    """
    if block_id not in self.blocks:
      raise ValueError(f"Cannot set entry block: Block ID '{block_id}' not found in CFG.")
    self.entry_block = self.blocks[block_id]

  def traverse_dfs(self, start_block: Optional[BasicBlock] = None) -> List[BasicBlock]:
    """Travers the CFG in Depth-First Search order.

    Args:
        start_block: The block to begin traversal from. Defaults to the entry block.

    Returns:
        A list of basic blocks in DFS visitation order.

    Raises:
        ValueError: If no start block is provided and the CFG has no entry block.
    """
    start = start_block if start_block is not None else self.entry_block
    if start is None:
      raise ValueError("Cannot traverse CFG: No entry block defined.")

    visited: Set[str] = set()
    result: List[BasicBlock] = []

    def dfs(block: BasicBlock) -> None:
      if block.id in visited:
        return
      visited.add(block.id)
      result.append(block)
      for succ in block.successors:
        dfs(succ)

    dfs(start)
    return result
