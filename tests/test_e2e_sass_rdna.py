"""Test suite for the e2e Sass and Rdna ODL targets."""

from typing import Dict, Union, Optional
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.core.compiler.ir import LogicalGraph


def test_odl_sass_rdna() -> None:
  """Verifies the behavior of ODL variants for hardware ISAs."""
  mgr: SemanticsManager = SemanticsManager()

  variant: Optional[Dict[str, Union[str, bool, int, float, list[str]]]] = mgr.resolve_variant(
    "conv2d", "sass"
  ) or mgr.resolve_variant("Conv2d", "sass")
  assert variant is not None
  assert variant["api"] == "Macro.Conv2d"

  variant_rdna: Optional[Dict[str, Union[str, bool, int, float, list[str]]]] = mgr.resolve_variant(
    "conv2d", "rdna"
  ) or mgr.resolve_variant("Conv2d", "rdna")
  assert variant_rdna is not None
  assert variant_rdna["api"] == "; Macro.Conv2d"

  variant_add: Optional[Dict[str, Union[str, bool, int, float, list[str]]]] = mgr.resolve_variant("Add", "sass")
  assert variant_add is not None
  assert variant_add["api"] == "FADD"


def test_sass_roundtrip_new_macros() -> None:
  """Verifies the roundtrip compilation of the new SASS macros."""
  from ml_switcheroo.core.compiler.backends.sass import SassBackend
  from ml_switcheroo.core.compiler.frontends.sass.parser import SassParser
  from ml_switcheroo.core.compiler.frontends.sass.lifter import SassLifter
  from ml_switcheroo.core.compiler.ir import LogicalNode

  # A dummy semantics manager that maps directly
  class DummySemantics:
    """A dummy semantics manager for testing."""

    def get_definition(self, kind: str) -> tuple[str, dict[str, str]]:
      """Gets a mock definition."""
      return (kind, {})

    def resolve_variant(self, aid: str, fw: str) -> Optional[Dict[str, str]]:
      """Resolves a mock variant."""
      if fw == "sass":
        if aid == "Abs":
          return {"api": "FABS"}
        return {"api": f"Macro.{aid}"}
      return None

  mgr: DummySemantics = DummySemantics()
  backend: SassBackend = SassBackend(mgr)
  lifter: SassLifter = SassLifter()

  macros_to_test: list[str] = ["Conv3d", "ReLU", "Flatten", "Reshape", "Mean"]

  for kind in macros_to_test:
    g_in: LogicalGraph = LogicalGraph(nodes=[LogicalNode("n1", kind, {"k": 3, "elements": 10})])
    sass_text: str = backend.compile(g_in)

    assert f"BEGIN {kind}" in sass_text, f"Missing BEGIN comment for {kind}"

    parser: SassParser = SassParser(sass_text)
    ast_nodes: list[str] = parser.parse().statements  # The parser statements are typed in other files, maybe ASTNodes
    g_out: LogicalGraph = lifter.lift(ast_nodes)

    assert len(g_out.nodes) == 1, f"Failed to lift {kind} correctly"
    assert g_out.nodes[0].kind == kind, f"Lifted node kind mismatch for {kind}"

  # Test Abs (1:1 opcode)
  g_in2: LogicalGraph = LogicalGraph(nodes=[LogicalNode("n1", "Abs")])
  sass_text2: str = backend.compile(g_in2)
  assert "FABS" in sass_text2

  parser2: SassParser = SassParser(sass_text2)
  ast_nodes2: list[str] = parser2.parse().statements
  g_out2: LogicalGraph = lifter.lift(ast_nodes2)

  # Because it's 1:1 without a BEGIN block, it parses as an assembly instruction node
  assert len(g_out2.nodes) == 1
  assert g_out2.nodes[0].kind == "asm.FABS"
