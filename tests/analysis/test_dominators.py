"""Test module for CFG Dominator analysis tools.

This module validates the correctness of graph theory algorithms implemented in the
`ml_switcheroo.analysis.dominators` package. It verifies the calculation of dominator
sets, immediate dominators (idoms), and the detection of back edges (loops) within
Control Flow Graphs.
"""

from __future__ import annotations

from ml_switcheroo.analysis.cfg import BasicBlock, ControlFlowGraph
from ml_switcheroo.analysis.dominators import build_dominator_sets, find_back_edges, find_immediate_dominators


def build_test_cfg() -> ControlFlowGraph:
  """Construct a standard non-trivial test CFG for dominator analysis.

  Constructs a CFG with the following structure:
  Entry: 1
  1 -> 2, 1 -> 5
  5 -> 2
  2 -> 3
  3 -> 4, 3 -> 6

  Returns:
      ControlFlowGraph: The populated Control Flow Graph.
  """
  cfg: ControlFlowGraph = ControlFlowGraph()
  # 1 -> 2 -> 3 -> 4
  # |    ^    |
  # v    |    v
  # 5 ---+    6
  b1: BasicBlock = cfg.get_or_create_block("1")
  b2: BasicBlock = cfg.get_or_create_block("2")
  b3: BasicBlock = cfg.get_or_create_block("3")
  b4: BasicBlock = cfg.get_or_create_block("4")
  b5: BasicBlock = cfg.get_or_create_block("5")
  b6: BasicBlock = cfg.get_or_create_block("6")

  b1.add_successor(b2)
  b1.add_successor(b5)

  b2.add_successor(b3)

  b3.add_successor(b4)
  b3.add_successor(b6)

  b5.add_successor(b2)

  cfg.set_entry_block("1")
  return cfg


def test_build_dominator_sets() -> None:
  """Test the calculation of complete dominator sets for all nodes in the CFG.

  Verifies that `build_dominator_sets` correctly implements the iterative data-flow
  equation for dominators. Asserts that the entry node dominates all reachable nodes,
  and that intermediate nodes have the correct strict dominators (e.g. {1, 2, 3} for node 3).
  """
  cfg: ControlFlowGraph = build_test_cfg()
  doms: dict[str, set[str]] = build_dominator_sets(cfg)

  # 1 dominates everything
  assert "1" in doms["1"]
  assert "1" in doms["2"]
  assert "1" in doms["3"]
  assert "1" in doms["4"]
  assert "1" in doms["5"]
  assert "1" in doms["6"]

  # 2 is dominated by 1 and 2
  assert doms["2"] == {"1", "2"}

  # 3 is dominated by 1, 2, 3
  assert doms["3"] == {"1", "2", "3"}

  # 5 is dominated by 1, 5
  assert doms["5"] == {"1", "5"}


def test_build_dominator_sets_empty_cfg() -> None:
  """Test dominator set calculation on an empty CFG.

  Verifies that calling the function on a CFG with no blocks safely returns
  an empty dictionary without error.
  """
  cfg: ControlFlowGraph = ControlFlowGraph()
  doms: dict[str, set[str]] = build_dominator_sets(cfg)
  assert doms == {}


def test_build_dominator_sets_unreachable() -> None:
  """Test dominator set behavior for unreachable CFG blocks.

  Verifies that blocks disconnected from the entry node fall back to safely
  dominating only themselves, preventing infinite loops or crashes during analysis.
  """
  cfg: ControlFlowGraph = ControlFlowGraph()
  cfg.get_or_create_block("1")
  cfg.get_or_create_block("2")  # unreachable
  doms: dict[str, set[str]] = build_dominator_sets(cfg)
  assert (
    "2" in doms["2"]
  )  # node 2 is unreachable, so doms initialization will eventually prune strictly to itself when updating if it has no predecessors


def test_find_immediate_dominators() -> None:
  """Test the identification of Immediate Dominators (idoms).

  Verifies that `find_immediate_dominators` correctly extracts the unique immediate
  dominator for each node from the full dominator sets. This is crucial for constructing
  the dominator tree.
  """
  cfg: ControlFlowGraph = build_test_cfg()
  doms: dict[str, set[str]] = build_dominator_sets(cfg)
  idoms: dict[str, str | None] = find_immediate_dominators(cfg, doms)

  assert idoms["1"] is None
  assert idoms["2"] == "1"
  assert idoms["3"] == "2"
  assert idoms["4"] == "3"
  assert idoms["5"] == "1"
  assert idoms["6"] == "3"


def test_find_back_edges() -> None:
  """Test the detection of back edges (loops) using dominator information.

  A back edge exists if a node has a successor that also dominates it.
  This test artificially adds a loop (`3 -> 2`) to the test CFG and verifies
  that `find_back_edges` accurately detects it.
  """
  cfg: ControlFlowGraph = build_test_cfg()
  # Add a loop 3 -> 2
  cfg.blocks["3"].add_successor(cfg.blocks["2"])

  doms: dict[str, set[str]] = build_dominator_sets(cfg)
  back_edges: list[tuple[str, str]] = find_back_edges(cfg, doms)

  assert len(back_edges) == 1
  assert back_edges[0] == ("3", "2")


def test_find_immediate_dominators_unreachable() -> None:
  """Test immediate dominator resolution for unreachable blocks.

  Verifies that an unreachable node correctly receives `None` as its immediate
  dominator, mirroring the behavior of the entry node.
  """
  cfg = ControlFlowGraph()
  cfg.get_or_create_block("1")
  cfg.get_or_create_block("2")
  cfg.set_entry_block("1")
  doms = build_dominator_sets(cfg)
  idoms = find_immediate_dominators(cfg, doms)
  assert idoms["2"] is None
