"""Dominator Tree analysis for Control Flow Graphs.

This module provides functions to calculate dominators, immediate dominators,
and back-edges for loop detection in CFGs.
"""

from typing import Dict, Set, List, Optional
from ml_switcheroo.analysis.cfg import ControlFlowGraph


def build_dominator_sets(cfg: ControlFlowGraph) -> Dict[str, Set[str]]:
  """Build the dominator sets for all blocks in the CFG.

  Block A dominates Block B if every path from the entry node to B must
  go through A. By definition, every block dominates itself.

  Args:
      cfg: The Control Flow Graph.

  Returns:
      A dictionary mapping block IDs to a set of block IDs that dominate them.
  """
  if cfg.entry_block is None:
    return {}

  all_nodes = set(cfg.blocks.keys())
  doms: Dict[str, Set[str]] = {node: set(all_nodes) for node in all_nodes}

  # Entry block is dominated only by itself
  doms[cfg.entry_block.id] = {cfg.entry_block.id}

  changed = True
  while changed:
    changed = False
    for node_id, node in cfg.blocks.items():
      if node_id == cfg.entry_block.id:
        continue

      # For all other nodes, Dom(n) = {n} U ( intersection over p in preds of Dom(p) )
      new_dom: Set[str] = set()
      if node.predecessors:
        new_dom = set(doms[node.predecessors[0].id])
        for pred in node.predecessors[1:]:
          new_dom = new_dom.intersection(doms[pred.id])

      new_dom.add(node_id)

      if new_dom != doms[node_id]:
        doms[node_id] = new_dom
        changed = True

  return doms


def find_immediate_dominators(cfg: ControlFlowGraph, doms: Dict[str, Set[str]]) -> Dict[str, Optional[str]]:
  """Calculate the immediate dominator for each block.

  The immediate dominator of node n (idom(n)) is the unique node that strictly
  dominates n but does not strictly dominate any other node that strictly dominates n.

  Args:
      cfg: The Control Flow Graph.
      doms: The dominator sets calculated by `build_dominator_sets`.

  Returns:
      A dictionary mapping block IDs to the ID of their immediate dominator.
      The entry block has an immediate dominator of None.
  """
  idoms: Dict[str, Optional[str]] = {}

  for node_id in cfg.blocks:
    if cfg.entry_block and node_id == cfg.entry_block.id:
      idoms[node_id] = None
      continue

    # Strict dominators of node_id (dominators excluding itself)
    strict_doms = sorted(list(doms[node_id] - {node_id}))

    idom = None
    for d in strict_doms:
      is_closest = True
      for other_d in strict_doms:
        if d != other_d and d in doms[other_d]:
          # d dominates other_d, meaning other_d is "closer" to node_id
          is_closest = False
          break
      if is_closest:
        idom = d
        break

    idoms[node_id] = idom

  return idoms


def find_back_edges(cfg: ControlFlowGraph, doms: Dict[str, Set[str]]) -> List[tuple[str, str]]:
  """Identify back-edges in the CFG to detect loops.

  A back-edge is an edge A -> B where B dominates A.

  Args:
      cfg: The Control Flow Graph.
      doms: The dominator sets calculated by `build_dominator_sets`.

  Returns:
      A list of tuples (source_id, target_id) representing the back-edges.
  """
  back_edges = []
  for source_id, block in cfg.blocks.items():
    for target in block.successors:
      target_id = target.id
      # If target dominates source, it's a back-edge
      if target_id in doms.get(source_id, set()):
        back_edges.append((source_id, target_id))
  return back_edges
