"""Test suite for the Rdna Roundtrip module."""

import typing
from unittest.mock import MagicMock

import pytest

from ml_switcheroo.core.compiler.backends.rdna import RdnaBackend
from ml_switcheroo.core.compiler.frontends.rdna.lifter import RdnaLifter
from ml_switcheroo.core.compiler.frontends.rdna.parser import RdnaParser
from ml_switcheroo.core.compiler.ir import LogicalEdge, LogicalGraph, LogicalNode
from ml_switcheroo.semantics.manager import SemanticsManager


@pytest.fixture
def semantics_mgr() -> SemanticsManager:
  """Execute semantics mgr helper logic.

  Returns:
      Computed test result or fixture value.
  """
  mgr: SemanticsManager = MagicMock(spec=SemanticsManager)

  def get_def(kind: str) -> typing.Optional[tuple[str, dict[str, typing.Any]]]:
    """Gets def.

    Args:
        kind: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    if kind == "Add":
      return ("Add", {})
    if "Conv2d" in kind:
      return ("Conv2d", {})
    return None

  def resolve_var(aid: str, fw: str) -> typing.Optional[dict[str, typing.Any]]:
    """Resolves variable.

    Args:
        aid: Test argument or fixture.
        fw: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    if fw == "rdna" and aid == "Add":
      return {"api": "v_add_f32"}
    return None

  setattr(mgr, "get_definition", MagicMock(side_effect=get_def))
  setattr(mgr, "resolve_variant", MagicMock(side_effect=resolve_var))
  return mgr


def test_rdna_roundtrip_macro(semantics_mgr: SemanticsManager) -> None:
  """Verifies the behavior of RDNA roundtrip macro.

  Args:
      semantics_mgr: Test argument or fixture.
  """
  nodes: typing.Any = {
    "img": LogicalNode("img", op_type="Input"),
    "conv": LogicalNode("conv", op_type="Conv2d", attributes={"k": 3}),
    "out": LogicalNode("out", op_type="Output"),
  }
  edges: typing.Any = [LogicalEdge("img", "conv"), LogicalEdge("conv", "out")]
  g_in: typing.Any = LogicalGraph(nodes=nodes, edges=edges)
  backend: typing.Any = RdnaBackend(semantics_mgr)
  rdna_text: str = backend.compile(g_in)
  assert "BEGIN Conv2d" in rdna_text
  assert "L_KY_conv" in rdna_text
  parser: typing.Any = RdnaParser(rdna_text)
  ast_nodes: list[typing.Any] = parser.parse().statements
  lifter: typing.Any = RdnaLifter()
  g_out: LogicalGraph = lifter.lift(ast_nodes)
  assert len(g_out.nodes) == 3
  node_ids: list[str] = list(g_out.nodes.keys())
  assert "img" in node_ids
  assert "conv" in node_ids
  assert "output" in node_ids
  conv_node: LogicalNode = g_out.nodes["conv"]
  assert conv_node.attributes["k"] == 3
