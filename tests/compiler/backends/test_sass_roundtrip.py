"""Test suite for the Sass Roundtrip module."""

import pytest
from unittest.mock import MagicMock
from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode, LogicalEdge
from ml_switcheroo.core.compiler.backends.sass import SassBackend
from ml_switcheroo.core.compiler.frontends.sass.parser import SassParser
from ml_switcheroo.core.compiler.frontends.sass.lifter import SassLifter
from ml_switcheroo.semantics.manager import SemanticsManager


@pytest.fixture
def semantics_mgr():
  """Provides a mock semantics mgr for testing."""
  mgr = MagicMock(spec=SemanticsManager)

  def get_def(kind):
    """Gets def."""
    if kind == "Add":
      return ("Add", {})
    if "Conv2d" in kind:
      return ("Conv2d", {})
    return None

  def resolve_var(aid, fw):
    """Resolves variable."""
    if fw == "sass" and aid == "Add":
      return {"api": "FADD"}
    return None

  mgr.get_definition.side_effect = get_def
  mgr.resolve_variant.side_effect = resolve_var
  return mgr


def test_round_trip_math_op(semantics_mgr):
  """Verifies the behavior of round trip math op."""
  g_in = LogicalGraph()
  g_in.nodes = [LogicalNode("x", "Input"), LogicalNode("y", "Input"), LogicalNode("z", "Add")]
  g_in.edges = [LogicalEdge("x", "z"), LogicalEdge("y", "z")]
  backend = SassBackend(semantics_mgr)
  sass_text = backend.compile(g_in)
  assert "FADD" in sass_text
  assert "Input x" in sass_text
  parser = SassParser(sass_text)
  ast_nodes = parser.parse()
  assert len(ast_nodes) > 0
  assert "FADD" in sass_text


def test_round_trip_macro_block(semantics_mgr):
  """Verifies the behavior of round trip macro block."""
  g_in = LogicalGraph()
  g_in.nodes = [LogicalNode("img", "Input"), LogicalNode("conv", "Conv2d", {"k": 3}), LogicalNode("out", "Output")]
  g_in.edges = [LogicalEdge("img", "conv"), LogicalEdge("conv", "out")]
  backend = SassBackend(semantics_mgr)
  sass_text = backend.compile(g_in)
  assert "BEGIN Conv2d" in sass_text
  assert "L_KY_conv" in sass_text
  parser = SassParser(sass_text)
  ast_nodes = parser.parse()
  lifter = SassLifter()
  g_out = lifter.lift(ast_nodes)
  assert len(g_out.nodes) == 3
  node_ids = [n.id for n in g_out.nodes]
  assert "img" in node_ids
  assert "conv" in node_ids
  assert "output" in node_ids
  conv_node = next((n for n in g_out.nodes if n.id == "conv"))
  assert conv_node.metadata["kernel_size"] == 3
