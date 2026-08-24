"""Test module."""

from ml_switcheroo.analysis.cfg import ControlFlowGraph
from ml_switcheroo.analysis.dominators import build_dominator_sets, find_immediate_dominators, find_back_edges


def build_test_cfg():
  """Test element."""
  cfg = ControlFlowGraph()
  # 1 -> 2 -> 3 -> 4
  # |    ^    |
  # v    |    v
  # 5 ---+    6
  b1 = cfg.get_or_create_block("1")
  b2 = cfg.get_or_create_block("2")
  b3 = cfg.get_or_create_block("3")
  b4 = cfg.get_or_create_block("4")
  b5 = cfg.get_or_create_block("5")
  b6 = cfg.get_or_create_block("6")

  b1.add_successor(b2)
  b1.add_successor(b5)

  b2.add_successor(b3)

  b3.add_successor(b4)
  b3.add_successor(b6)

  b5.add_successor(b2)

  cfg.set_entry_block("1")
  return cfg


def test_build_dominator_sets():
  """Test element."""
  cfg = build_test_cfg()
  doms = build_dominator_sets(cfg)

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


def test_build_dominator_sets_empty_cfg():
  """Test element."""
  cfg = ControlFlowGraph()
  doms = build_dominator_sets(cfg)
  assert doms == {}


def test_build_dominator_sets_unreachable():
  """Test element."""
  cfg = ControlFlowGraph()
  cfg.get_or_create_block("1")
  cfg.get_or_create_block("2")  # unreachable
  doms = build_dominator_sets(cfg)
  assert (
    "2" in doms["2"]
  )  # node 2 is unreachable, so doms initialization will eventually prune strictly to itself when updating if it has no predecessors


def test_find_immediate_dominators():
  """Test element."""
  cfg = build_test_cfg()
  doms = build_dominator_sets(cfg)
  idoms = find_immediate_dominators(cfg, doms)

  assert idoms["1"] is None
  assert idoms["2"] == "1"
  assert idoms["3"] == "2"
  assert idoms["4"] == "3"
  assert idoms["5"] == "1"
  assert idoms["6"] == "3"


def test_find_back_edges():
  """Test element."""
  cfg = build_test_cfg()
  # Add a loop 3 -> 2
  cfg.blocks["3"].add_successor(cfg.blocks["2"])

  doms = build_dominator_sets(cfg)
  back_edges = find_back_edges(cfg, doms)

  assert len(back_edges) == 1
  assert back_edges[0] == ("3", "2")
