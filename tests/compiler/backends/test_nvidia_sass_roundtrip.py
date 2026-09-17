"""Test suite for the Sass Roundtrip module."""

import typing
from unittest.mock import MagicMock

import pytest

from ml_switcheroo.core.compiler.backends.nvidia_sass import NvidiaSassBackend
from ml_switcheroo.core.compiler.frontends.nvidia_sass.lifter import NvidiaSassLifter
from ml_switcheroo.core.compiler.frontends.nvidia_sass.parser import NvidiaSassParser
from ml_switcheroo.core.compiler.ir import LogicalEdge, LogicalGraph, LogicalNode
from ml_switcheroo.semantics.manager import SemanticsManager


@pytest.fixture
def semantics_mgr() -> SemanticsManager:
  """Docstring."""
  mgr: SemanticsManager = MagicMock(spec=SemanticsManager)

  def get_def(kind: str) -> typing.Optional[tuple[str, dict[str, typing.Any]]]:
    """Gets def."""
    if kind == "Add":
      return ("Add", {})
    if "Conv2d" in kind:
      return ("Conv2d", {})
    return None

  def resolve_var(aid: str, fw: str) -> typing.Optional[dict[str, typing.Any]]:
    """Resolves variable."""
    if fw == "nvidia_sass" and aid == "Add":
      return {"api": "FADD"}
    return None

  mgr.get_definition = MagicMock(side_effect=get_def)
  mgr.resolve_variant = MagicMock(side_effect=resolve_var)
  return mgr


def test_round_trip_math_op(semantics_mgr: SemanticsManager) -> None:
  """Verifies the behavior of round trip math op."""
  nodes = {
    "x": LogicalNode("x", op_type="Input"),
    "y": LogicalNode("y", op_type="Input"),
    "z": LogicalNode("z", op_type="Add"),
  }
  edges = [LogicalEdge("x", "z"), LogicalEdge("y", "z")]
  g_in = LogicalGraph(nodes=nodes, edges=edges)
  backend = NvidiaSassBackend(semantics_mgr)
  sass_text: str = backend.compile(g_in)
  assert "FADD" in sass_text
  assert "Input x" in sass_text
  parser = NvidiaSassParser(sass_text)
  ast_nodes: list[typing.Any] = parser.parse().statements
  assert len(ast_nodes) > 0
  assert "FADD" in sass_text


def test_round_trip_macro_block(semantics_mgr: SemanticsManager) -> None:
  """Verifies the behavior of round trip macro block."""
  nodes = {
    "img": LogicalNode("img", op_type="Input"),
    "conv": LogicalNode("conv", op_type="Conv2d", attributes={"k": 3}),
    "out": LogicalNode("out", op_type="Output"),
  }
  edges = [LogicalEdge("img", "conv"), LogicalEdge("conv", "out")]
  g_in = LogicalGraph(nodes=nodes, edges=edges)
  backend = NvidiaSassBackend(semantics_mgr)
  sass_text: str = backend.compile(g_in)
  assert "BEGIN Conv2d" in sass_text
  assert "L_KY_conv" in sass_text
  parser = NvidiaSassParser(sass_text)
  ast_nodes: list[typing.Any] = parser.parse().statements
  lifter = NvidiaSassLifter()
  g_out: LogicalGraph = lifter.lift(ast_nodes)
  assert len(g_out.nodes) == 3
  node_ids: list[str] = list(g_out.nodes.keys())
  assert "img" in node_ids
  assert "conv" in node_ids
  assert "output" in node_ids
  conv_node: LogicalNode = g_out.nodes["conv"]
  assert conv_node.attributes["kernel_size"] == 3
