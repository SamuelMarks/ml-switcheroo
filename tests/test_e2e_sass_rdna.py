"""Test suite for the e2e Sass and Rdna ODL targets."""

from ml_switcheroo.semantics.manager import SemanticsManager


def test_odl_sass_rdna() -> None:
  """Verifies the behavior of ODL variants for hardware ISAs."""
  mgr = SemanticsManager()

  variant = mgr.resolve_variant("conv2d", "sass") or mgr.resolve_variant("Conv2d", "sass")
  assert variant is not None
  assert variant["api"] == "Macro.Conv2d"

  variant_rdna = mgr.resolve_variant("conv2d", "rdna") or mgr.resolve_variant("Conv2d", "rdna")
  assert variant_rdna is not None
  assert variant_rdna["api"] == "; Macro.Conv2d"

  variant_add = mgr.resolve_variant("Add", "sass")
  assert variant_add is not None
  assert variant_add["api"] == "FADD"


def test_sass_roundtrip_new_macros() -> None:
  """Verifies the roundtrip compilation of the new SASS macros."""
  from ml_switcheroo.core.compiler.backends.sass import SassBackend
  from ml_switcheroo.core.compiler.frontends.sass.parser import SassParser
  from ml_switcheroo.core.compiler.frontends.sass.lifter import SassLifter
  from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode

  # A dummy semantics manager that maps directly
  class DummySemantics:
    def get_definition(self, kind):
      return (kind, {})

    def resolve_variant(self, aid, fw):
      if fw == "sass":
        # For Abs we use FABS, others use Macro.<name> if they are macros
        if aid == "Abs":
          return {"api": "FABS"}
        if aid == "Conv3d":
          return {"api": "Macro.Conv3d"}
        if aid == "ReLU":
          return {"api": "Macro.ReLU"}
        if aid == "Flatten":
          return {"api": "Macro.Flatten"}
        if aid == "Reshape":
          return {"api": "Macro.Reshape"}
        if aid == "Mean":
          return {"api": "Macro.Mean"}
      return None

  mgr = DummySemantics()
  backend = SassBackend(mgr)
  lifter = SassLifter()

  macros_to_test = ["Conv3d", "ReLU", "Flatten", "Reshape", "Mean"]

  for kind in macros_to_test:
    g_in = LogicalGraph(nodes=[LogicalNode("n1", kind, {"k": 3, "elements": 10})])
    sass_text = backend.compile(g_in)

    assert f"BEGIN {kind}" in sass_text, f"Missing BEGIN comment for {kind}"

    parser = SassParser(sass_text)
    ast_nodes = parser.parse().statements
    g_out = lifter.lift(ast_nodes)

    assert len(g_out.nodes) == 1, f"Failed to lift {kind} correctly"
    assert g_out.nodes[0].kind == kind, f"Lifted node kind mismatch for {kind}"

  # Test Abs (1:1 opcode)
  g_in = LogicalGraph(nodes=[LogicalNode("n1", "Abs")])
  sass_text = backend.compile(g_in)
  assert "FABS" in sass_text

  parser = SassParser(sass_text)
  ast_nodes = parser.parse().statements
  g_out = lifter.lift(ast_nodes)

  # Because it's 1:1 without a BEGIN block, it parses as an assembly instruction node
  assert len(g_out.nodes) == 1
  assert g_out.nodes[0].kind == "asm.FABS"
