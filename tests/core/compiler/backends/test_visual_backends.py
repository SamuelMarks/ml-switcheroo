"""Unit tests for `TikzBackend` and `LatexBackend` (`ml_switcheroo.core.compiler.backends.visual_backends`)."""

import pytest

from ml_switcheroo.core.compiler.backends.visual_backends import LatexBackend, TikzBackend
from ml_switcheroo.core.compiler.backends.visual_latex import LatexBackend as VL_LatexBackend
from ml_switcheroo.core.compiler.backends.visual_tikz import TikzBackend as VT_TikzBackend
from ml_switcheroo.core.compiler.ir import LogicalEdge, LogicalGraph, LogicalNode


def _sample_graph() -> LogicalGraph:
  """Build a sample `LogicalGraph` spanning `Input`, `Linear`, `Add`, `StateOp`, `MemoryOp`, and `Output`.

  Returns:
      Populated `LogicalGraph`.
  """
  nodes = {
    "in1": LogicalNode("in1", "Input", attributes={"name": "x"}),
    "l1": LogicalNode("l1", "Linear", attributes={"leading_trivia": "# cmt", "arg_0": 10, "shape": "(10, 10)"}),
    "func_add": LogicalNode("func_add", "func_add", attributes={"leading_trivia": "# cmt", "arg_1": "1"}),
    "func_abs": LogicalNode("func_abs", "torch.abs", attributes={"arg_x": "2", "other": "3"}),
    "state1": LogicalNode("state1", "StateOp"),
    "mem1": LogicalNode("mem1", "MemoryOp"),
    "Output": LogicalNode("Output", "Output"),
  }
  return LogicalGraph(
    name="TestGraph",
    nodes=nodes,
    edges=[
      LogicalEdge("in1", "l1"),
      LogicalEdge("l1", "func_add"),
      LogicalEdge("func_add", "func_abs"),
      LogicalEdge("func_abs", "state1"),
      LogicalEdge("state1", "mem1"),
      LogicalEdge("mem1", "Output"),
      LogicalEdge("in1", "l1"),
    ],
  )


def test_tikz_backend_nodes_cycles_disconnected_and_ranks() -> None:
  """Verify `TikzBackend` and `VT_TikzBackend` across normal, cyclic, disconnected, and empty graphs."""
  tb = TikzBackend()
  vtb = VT_TikzBackend()

  stateful_tn = tb._create_tikz_node(LogicalNode("L1", "Linear", attributes={"arg_0": 10}), 0, 0)
  assert stateful_tn.node_id == "L1"
  assert not isinstance(stateful_tn.content, str)
  assert "Linear" in stateful_tn.content.emit()
  assert r"arg\_0: 10" in stateful_tn.content.emit()
  assert tb._create_tikz_node(LogicalNode("R1", "relu"), 0, 0).node_id == "R1"
  assert tb._create_tikz_node(LogicalNode("in", "Input"), 0, 0).node_id == "in"

  for backend in (tb, vtb):
    assert "tikzpicture" in backend.compile(_sample_graph())
    assert "tikzpicture" in backend.compile(LogicalGraph())

    # Truncated >3 attributes + duplicate edges + disconnected + pure cycle
    many_attrs = LogicalGraph(
      nodes={
        "in1": LogicalNode("in1", "Input"),
        "u1": LogicalNode("u1", "Unknown", attributes={"k": "1", "v": "2", "b": "3", "c": "4"}),
        "u2": LogicalNode("u2", "Unknown"),
      },
      edges=[LogicalEdge("in1", "u1"), LogicalEdge("in1", "u1")],
    )
    assert "tikzpicture" in backend.compile(many_attrs)

    mixed_cycle = LogicalGraph(
      nodes={"n1": LogicalNode("n1", "Input"), "c1": LogicalNode("c1", "Op"), "c2": LogicalNode("c2", "Op")},
      edges=[LogicalEdge("c1", "c2"), LogicalEdge("c2", "c1")],
    )
    assert "tikzpicture" in backend.compile(mixed_cycle)

    pure_cycle = LogicalGraph(
      nodes={"A": LogicalNode("A", "Op"), "B": LogicalNode("B", "Op")},
      edges=[LogicalEdge("A", "B"), LogicalEdge("B", "A")],
    )
    assert "tikzpicture" in backend.compile(pure_cycle)

  # Multi-input fan-in rank centering
  fan_in = LogicalGraph(
    "FanIn",
    nodes={"A": LogicalNode("A", "Input"), "B": LogicalNode("B", "Input"), "C": LogicalNode("C", "Linear")},
    edges=[LogicalEdge("A", "C"), LogicalEdge("B", "C")],
  )
  assert tb._calculate_layout(fan_in) == {"A": (-1.5, 0.0), "B": (1.5, 0.0), "C": (0.0, -2.5)}


def test_latex_backend_compilation_fallbacks_and_cycles(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `LatexBackend` and `VL_LatexBackend` across named/unnamed, outputless, cyclic, and pending graphs.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  lb = LatexBackend()
  vlb = VL_LatexBackend()

  for backend in (lb, vlb):
    doc = backend.compile(_sample_graph())
    assert r"\begin{DefModel}{TestGraph}" in doc
    assert "Linear" in doc
    assert "Add" in doc

    assert "Model" in backend.compile(LogicalGraph())
    assert "Model" in backend.compile(LogicalGraph(nodes={"op1": LogicalNode("op1", "Op")}))

    no_out = LogicalGraph(
      nodes={"in1": LogicalNode("in1", "Input"), "op1": LogicalNode("op1", "some_mod.foo")},
      edges=[LogicalEdge("in1", "op1")],
    )
    assert "last_step" in backend.compile(no_out)

  # Cyclic functional nodes and `_pending_edges` with missing target node data
  cyc = LogicalGraph(
    "T",
    nodes={
      "n1": LogicalNode("n1", "Input"),
      "func_n2": LogicalNode("func_n2", "foo.bar", attributes={"notkey": "val"}),
      "n3": LogicalNode("n3", "Output"),
      "func_n4": LogicalNode("func_n4", "Op"),
    },
    edges=[
      LogicalEdge("n1", "func_n2"),
      LogicalEdge("func_n2", "n3"),
      LogicalEdge("func_n4", "func_n2"),
      LogicalEdge("func_n2", "func_n4"),
    ],
  )
  assert len(lb._transcode_graph(cyc, "T").children) > 0

  pending = LogicalGraph("Custom", nodes={}, edges=[LogicalEdge("in", "target")])
  monkeypatch.setattr(pending, "_pending_edges", [LogicalEdge("in", "target")], raising=False)
  assert "op_target" in lb.compile(pending)
