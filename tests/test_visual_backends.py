"""Docstring."""

from ml_switcheroo.core.compiler.backends.visual_backends import LatexBackend as VB_LatexBackend
from ml_switcheroo.core.compiler.backends.visual_backends import TikzBackend as VB_TikzBackend
from ml_switcheroo.core.compiler.backends.visual_latex import LatexBackend as VL_LatexBackend
from ml_switcheroo.core.compiler.backends.visual_tikz import TikzBackend as VT_TikzBackend
from ml_switcheroo.core.compiler.ir import LogicalEdge, LogicalGraph, LogicalNode


def get_dummy_graph() -> LogicalGraph:
  """Docstring."""
  nodes = {
    "in1": LogicalNode(id="in1", op_type="Input"),
    "l1": LogicalNode(id="l1", op_type="Linear", attributes={"k": "3", "shape": "(10, 10)"}),
    "add1": LogicalNode(id="add1", op_type="Add"),
    "state1": LogicalNode(id="state1", op_type="StateOp"),
    "mem1": LogicalNode(id="mem1", op_type="MemoryOp"),
    "Output": LogicalNode(id="Output", op_type="Output"),
  }
  return LogicalGraph(
    name="TestGraph",
    nodes=nodes,
    edges=[
      LogicalEdge("in1", "l1"),
      LogicalEdge("l1", "add1"),
      LogicalEdge("add1", "state1"),
      LogicalEdge("state1", "mem1"),
      LogicalEdge("mem1", "Output"),
    ],
  )


def test_vb_tikz_backend() -> None:
  """Docstring."""
  backend: VB_TikzBackend = VB_TikzBackend()
  code: str = backend.compile(get_dummy_graph())
  assert "tikzpicture" in code


def test_vt_tikz_backend() -> None:
  """Docstring."""
  backend: VT_TikzBackend = VT_TikzBackend()
  code: str = backend.compile(get_dummy_graph())
  assert "tikzpicture" in code


def test_vb_latex_backend() -> None:
  """Docstring."""
  backend: VB_LatexBackend = VB_LatexBackend()
  code: str = backend.compile(get_dummy_graph())
  assert "documentclass" in code


def test_vl_latex_backend() -> None:
  """Docstring."""
  backend: VL_LatexBackend = VL_LatexBackend()
  code: str = backend.compile(get_dummy_graph())
  assert "documentclass" in code


def test_vb_tikz_backend_metadata_edgecases() -> None:
  """Docstring."""
  nodes = {
    "in1": LogicalNode(id="in1", op_type="Input"),
    "unkn1": LogicalNode(id="unkn1", op_type="Unknown", attributes={"k": "val", "v": "val2", "b": "v3", "c": "v4"}),
    "unkn2": LogicalNode(id="unkn2", op_type="Unknown", attributes={"k": "val", "v": "val2", "b": "v3", "c": "v4"}),
  }
  graph: LogicalGraph = LogicalGraph(
    nodes=nodes,
    edges=[
      LogicalEdge("in1", "unkn1"),
      LogicalEdge("in1", "unkn1"),
    ],  # duplicate edge to trigger `target_id in visited_ops`
  )
  backend: VB_TikzBackend = VB_TikzBackend()
  code: str = backend.compile(graph)
  assert "tikzpicture" in code

  backend_vt: VT_TikzBackend = VT_TikzBackend()
  code_vt: str = backend_vt.compile(graph)
  assert "tikzpicture" in code_vt


def test_vt_tikz_backend_metadata_edgecases() -> None:
  """Docstring."""
  nodes = {
    "in1": LogicalNode(id="in1", op_type="Input"),
    "unkn1": LogicalNode(id="unkn1", op_type="Unknown", attributes={"k": "val", "v": "val2", "b": "v3", "c": "v4"}),
  }
  graph: LogicalGraph = LogicalGraph(
    nodes=nodes,
    edges=[LogicalEdge("in1", "unkn1")],
  )
  backend: VT_TikzBackend = VT_TikzBackend()
  code: str = backend.compile(graph)
  assert "tikzpicture" in code


def test_layout_cycles_and_disconnected() -> None:
  """Docstring."""
  nodes = {
    "A": LogicalNode(id="A", op_type="Op"),
    "B": LogicalNode(id="B", op_type="Op"),
    "C": LogicalNode(id="C", op_type="Op"),
  }
  graph: LogicalGraph = LogicalGraph(
    nodes=nodes,
    edges=[
      LogicalEdge("A", "B"),
      LogicalEdge("B", "A"),  # cycle
    ],
  )
  # A and B are in a cycle, C is disconnected
  backend: VB_TikzBackend = VB_TikzBackend()
  code: str = backend.compile(graph)
  assert "tikzpicture" in code

  # Pure cycle to trigger `if not queue`
  nodes_cycle = {
    "A": LogicalNode(id="A", op_type="Op"),
    "B": LogicalNode(id="B", op_type="Op"),
  }
  graph_cycle: LogicalGraph = LogicalGraph(
    nodes=nodes_cycle,
    edges=[
      LogicalEdge("A", "B"),
      LogicalEdge("B", "A"),  # cycle
    ],
  )
  backend.compile(graph_cycle)
  backend_vt: VT_TikzBackend = VT_TikzBackend()
  backend_vt.compile(graph_cycle)

  # Empty graph
  code2: str = backend.compile(LogicalGraph(nodes={}, edges=[]))
  assert "tikzpicture" in code2


def test_latex_edgecases() -> None:
  """Docstring."""
  nodes = {
    "in1": LogicalNode(id="in1", op_type="Input"),
    "func_add": LogicalNode(id="func_add", op_type="func_add", attributes={"arg_1": "1"}),
    "func_abs": LogicalNode(id="func_abs", op_type="torch.abs", attributes={"arg_something": "2", "other": "3"}),
    "state": LogicalNode(id="state", op_type="StateOp", attributes={}),
    "Output": LogicalNode(id="Output", op_type="Output"),
    "unkn2": LogicalNode(id="unkn2", op_type="Unknown", attributes={"k": "val", "v": "val2", "b": "v3", "c": "v4"}),
  }
  graph: LogicalGraph = LogicalGraph(
    nodes=nodes,
    edges=[
      LogicalEdge("in1", "func_add"),
      LogicalEdge("func_add", "func_abs"),
      LogicalEdge("func_abs", "state"),
      LogicalEdge("state", "Output"),
      LogicalEdge("in1", "func_add"),  # duplicate to hit visited_ops
    ],
  )
  backend: VB_LatexBackend = VB_LatexBackend()
  code: str = backend.compile(graph)
  assert "Add" in code
  assert "Abs" in code
  assert "StateOp" in code

  backend2: VL_LatexBackend = VL_LatexBackend()
  code2: str = backend2.compile(graph)
  assert "Add" in code2


def test_latex_no_input() -> None:
  """Docstring."""
  nodes = {
    "op1": LogicalNode(id="op1", op_type="Op"),
  }
  graph: LogicalGraph = LogicalGraph(
    nodes=nodes,
    edges=[],
  )
  backend: VB_LatexBackend = VB_LatexBackend()
  code: str = backend.compile(graph)
  assert "documentclass" in code

  backend2: VL_LatexBackend = VL_LatexBackend()
  backend2.compile(graph)


def test_latex_no_output() -> None:
  """Docstring."""
  nodes = {
    "in1": LogicalNode(id="in1", op_type="Input"),
    "op1": LogicalNode(id="op1", op_type="Op"),
  }
  graph: LogicalGraph = LogicalGraph(
    nodes=nodes,
    edges=[LogicalEdge("in1", "op1")],
  )
  backend: VB_LatexBackend = VB_LatexBackend()
  code: str = backend.compile(graph)
  assert "last_step" in code

  backend2: VL_LatexBackend = VL_LatexBackend()
  code2: str = backend2.compile(graph)
  assert "last_step" in code2


def test_tikz_empty_and_disconnected_vt() -> None:
  """Docstring."""
  nodes = {
    "A": LogicalNode(id="A", op_type="Op"),
    "B": LogicalNode(id="B", op_type="Op"),
    "C": LogicalNode(id="C", op_type="Op"),
  }
  graph: LogicalGraph = LogicalGraph(
    nodes=nodes,
    edges=[
      LogicalEdge("A", "B"),
      LogicalEdge("B", "A"),  # cycle
    ],
  )
  backend_vt: VT_TikzBackend = VT_TikzBackend()
  code: str = backend_vt.compile(graph)
  assert "tikzpicture" in code

  code2: str = backend_vt.compile(LogicalGraph(nodes={}, edges=[]))
  assert "tikzpicture" in code2
