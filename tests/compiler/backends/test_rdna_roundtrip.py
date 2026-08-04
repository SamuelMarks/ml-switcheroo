"""Test suite for the Rdna Roundtrip module."""

import pytest
from unittest.mock import MagicMock
from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode, LogicalEdge
from ml_switcheroo.core.compiler.backends.rdna import RdnaBackend
from ml_switcheroo.core.compiler.frontends.rdna.parser import RdnaParser
from ml_switcheroo.core.compiler.frontends.rdna.lifter import RdnaLifter
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
    if fw == "rdna" and aid == "Add":
      return {"api": "v_add_f32"}
    return None

  mgr.get_definition.side_effect = get_def
  mgr.resolve_variant.side_effect = resolve_var
  return mgr


def test_rdna_roundtrip_macro(semantics_mgr):
  """Verifies the behavior of RDNA roundtrip macro."""
  g_in = LogicalGraph()
  g_in.nodes = [LogicalNode("img", "Input"), LogicalNode("conv", "Conv2d", {"k": 3}), LogicalNode("out", "Output")]
  g_in.edges = [LogicalEdge("img", "conv"), LogicalEdge("conv", "out")]
  backend = RdnaBackend(semantics_mgr)
  rdna_text = backend.compile(g_in)
  assert "BEGIN Conv2d" in rdna_text
  assert "L_KY_conv" in rdna_text
  parser = RdnaParser(rdna_text)
  ast_nodes = parser.parse().statements
  lifter = RdnaLifter()
  g_out = lifter.lift(ast_nodes)
  assert len(g_out.nodes) == 3
  node_ids = [n.id for n in g_out.nodes]
  assert "img" in node_ids
  assert "conv" in node_ids
  assert "output" in node_ids
  conv_node = next((n for n in g_out.nodes if n.id == "conv"))
  assert conv_node.metadata["k"] == 3
