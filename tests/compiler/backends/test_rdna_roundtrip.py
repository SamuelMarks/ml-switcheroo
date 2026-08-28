"""Test suite for the Rdna Roundtrip module."""

import pytest
import typing
from unittest.mock import MagicMock
from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode, LogicalEdge
from ml_switcheroo.core.compiler.backends.rdna import RdnaBackend
from ml_switcheroo.core.compiler.frontends.rdna.parser import RdnaParser
from ml_switcheroo.core.compiler.frontends.rdna.lifter import RdnaLifter
from ml_switcheroo.semantics.manager import SemanticsManager


@pytest.fixture
def semantics_mgr() -> SemanticsManager:
  """Provides a mock semantics mgr for testing."""
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
    if fw == "rdna" and aid == "Add":
      return {"api": "v_add_f32"}
    return None

  mgr.get_definition = MagicMock(side_effect=get_def)
  mgr.resolve_variant = MagicMock(side_effect=resolve_var)
  return mgr


def test_rdna_roundtrip_macro(semantics_mgr: SemanticsManager) -> None:
  """Verifies the behavior of RDNA roundtrip macro."""
  g_in = LogicalGraph()
  g_in.nodes = [LogicalNode("img", "Input"), LogicalNode("conv", "Conv2d", {"k": 3}), LogicalNode("out", "Output")]
  g_in.edges = [LogicalEdge("img", "conv"), LogicalEdge("conv", "out")]
  backend = RdnaBackend(semantics_mgr)
  rdna_text: str = backend.compile(g_in)
  assert "BEGIN Conv2d" in rdna_text
  assert "L_KY_conv" in rdna_text
  parser = RdnaParser(rdna_text)
  ast_nodes: list[typing.Any] = parser.parse().statements
  lifter = RdnaLifter()
  g_out: LogicalGraph = lifter.lift(ast_nodes)
  assert len(g_out.nodes) == 3
  node_ids: list[str] = [n.id for n in g_out.nodes]
  assert "img" in node_ids
  assert "conv" in node_ids
  assert "output" in node_ids
  conv_node: LogicalNode = next((n for n in g_out.nodes if n.id == "conv"))
  assert conv_node.metadata["k"] == 3
