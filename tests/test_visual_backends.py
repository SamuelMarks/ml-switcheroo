"""Docstring."""

from ml_switcheroo.core.compiler.backends.visual_backends import (
  TikzBackend as VB_TikzBackend,
  LatexBackend as VB_LatexBackend,
)
from ml_switcheroo.core.compiler.backends.visual_tikz import TikzBackend as VT_TikzBackend
from ml_switcheroo.core.compiler.backends.visual_latex import LatexBackend as VL_LatexBackend
from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode, LogicalEdge


def get_dummy_graph():
  """Docstring."""
  return LogicalGraph(
    name="TestGraph",
    nodes=[
      LogicalNode(id="in1", kind="Input"),
      LogicalNode(id="l1", kind="Linear", metadata={"k": "3", "shape": "(10, 10)"}),
      LogicalNode(id="add1", kind="Add"),
      LogicalNode(id="state1", kind="StateOp"),
      LogicalNode(id="mem1", kind="MemoryOp"),
      LogicalNode(id="Output", kind="Output"),
    ],
    edges=[
      LogicalEdge("in1", "l1"),
      LogicalEdge("l1", "add1"),
      LogicalEdge("add1", "state1"),
      LogicalEdge("state1", "mem1"),
      LogicalEdge("mem1", "Output"),
    ],
  )


def test_vb_tikz_backend():
  """Docstring."""
  backend = VB_TikzBackend()
  code = backend.compile(get_dummy_graph())
  assert "tikzpicture" in code


def test_vt_tikz_backend():
  """Docstring."""
  backend = VT_TikzBackend()
  code = backend.compile(get_dummy_graph())
  assert "tikzpicture" in code


def test_vb_latex_backend():
  """Docstring."""
  backend = VB_LatexBackend()
  code = backend.compile(get_dummy_graph())
  assert "documentclass" in code


def test_vl_latex_backend():
  """Docstring."""
  backend = VL_LatexBackend()
  code = backend.compile(get_dummy_graph())
  assert "documentclass" in code


def test_vb_tikz_backend_metadata_edgecases():
  """Docstring."""
  graph = LogicalGraph(
    nodes=[
      LogicalNode(id="in1", kind="Input"),
      LogicalNode(id="unkn1", kind="Unknown", metadata={"k": "val", "v": "val2", "b": "v3", "c": "v4"}),
      LogicalNode(id="unkn2", kind="Unknown", metadata={"k": "val", "v": "val2", "b": "v3", "c": "v4"}),
    ],
    edges=[
      LogicalEdge("in1", "unkn1"),
      LogicalEdge("in1", "unkn1"),
    ],  # duplicate edge to trigger `target_id in visited_ops`
  )
  backend = VB_TikzBackend()
  code = backend.compile(graph)
  assert "tikzpicture" in code

  backend_vt = VT_TikzBackend()
  code_vt = backend_vt.compile(graph)
  assert "tikzpicture" in code_vt


def test_vt_tikz_backend_metadata_edgecases():
  """Docstring."""
  graph = LogicalGraph(
    nodes=[
      LogicalNode(id="in1", kind="Input"),
      LogicalNode(id="unkn1", kind="Unknown", metadata={"k": "val", "v": "val2", "b": "v3", "c": "v4"}),
    ],
    edges=[LogicalEdge("in1", "unkn1")],
  )
  backend = VT_TikzBackend()
  code = backend.compile(graph)
  assert "tikzpicture" in code


def test_layout_cycles_and_disconnected():
  """Docstring."""
  graph = LogicalGraph(
    nodes=[
      LogicalNode(id="A", kind="Op"),
      LogicalNode(id="B", kind="Op"),
      LogicalNode(id="C", kind="Op"),
    ],
    edges=[
      LogicalEdge("A", "B"),
      LogicalEdge("B", "A"),  # cycle
    ],
  )
  # A and B are in a cycle, C is disconnected
  backend = VB_TikzBackend()
  code = backend.compile(graph)
  assert "tikzpicture" in code

  # Pure cycle to trigger `if not queue`
  graph_cycle = LogicalGraph(
    nodes=[
      LogicalNode(id="A", kind="Op"),
      LogicalNode(id="B", kind="Op"),
    ],
    edges=[
      LogicalEdge("A", "B"),
      LogicalEdge("B", "A"),  # cycle
    ],
  )
  backend.compile(graph_cycle)
  backend_vt = VT_TikzBackend()
  backend_vt.compile(graph_cycle)

  # Empty graph
  code2 = backend.compile(LogicalGraph(nodes=[], edges=[]))
  assert "tikzpicture" in code2


def test_latex_edgecases():
  """Docstring."""
  graph = LogicalGraph(
    nodes=[
      LogicalNode(id="in1", kind="Input"),
      LogicalNode(id="func_add", kind="func_add", metadata={"arg_1": "1"}),
      LogicalNode(id="func_abs", kind="torch.abs", metadata={"arg_something": "2", "other": "3"}),
      LogicalNode(id="state", kind="StateOp", metadata={}),
      LogicalNode(id="Output", kind="Output"),
      LogicalNode(id="unkn2", kind="Unknown", metadata={"k": "val", "v": "val2", "b": "v3", "c": "v4"}),
    ],
    edges=[
      LogicalEdge("in1", "func_add"),
      LogicalEdge("func_add", "func_abs"),
      LogicalEdge("func_abs", "state"),
      LogicalEdge("state", "Output"),
      LogicalEdge("in1", "func_add"),  # duplicate to hit visited_ops
    ],
  )
  backend = VB_LatexBackend()
  code = backend.compile(graph)
  assert "Add" in code
  assert "Abs" in code
  assert "StateOp" in code

  backend2 = VL_LatexBackend()
  code2 = backend2.compile(graph)
  assert "Add" in code2


def test_latex_no_input():
  """Docstring."""
  graph = LogicalGraph(
    nodes=[
      LogicalNode(id="op1", kind="Op"),
    ],
    edges=[],
  )
  backend = VB_LatexBackend()
  code = backend.compile(graph)
  assert "documentclass" in code

  backend2 = VL_LatexBackend()
  backend2.compile(graph)


def test_latex_no_output():
  """Docstring."""
  graph = LogicalGraph(
    nodes=[
      LogicalNode(id="in1", kind="Input"),
      LogicalNode(id="op1", kind="Op"),
    ],
    edges=[LogicalEdge("in1", "op1")],
  )
  backend = VB_LatexBackend()
  code = backend.compile(graph)
  assert "last_step" in code

  backend2 = VL_LatexBackend()
  code2 = backend2.compile(graph)
  assert "last_step" in code2


def test_tikz_empty_and_disconnected_vt():
  """Docstring."""
  graph = LogicalGraph(
    nodes=[
      LogicalNode(id="A", kind="Op"),
      LogicalNode(id="B", kind="Op"),
      LogicalNode(id="C", kind="Op"),
    ],
    edges=[
      LogicalEdge("A", "B"),
      LogicalEdge("B", "A"),  # cycle
    ],
  )
  backend_vt = VT_TikzBackend()
  code = backend_vt.compile(graph)
  assert "tikzpicture" in code

  code2 = backend_vt.compile(LogicalGraph(nodes=[], edges=[]))
  assert "tikzpicture" in code2
