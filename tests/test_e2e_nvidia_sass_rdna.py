"""Test suite for the e2e Sass and Rdna ODL targets."""

from typing import Dict, Optional, Union

from ml_switcheroo.core.compiler.ir import LogicalGraph
from ml_switcheroo.semantics.manager import SemanticsManager


def test_odl_sass_rdna() -> None:
  """Verifies the behavior of ODL variants for hardware ISAs."""
  mgr: SemanticsManager = SemanticsManager()

  variant: Optional[Dict[str, Union[str, bool, int, float, list[str]]]] = mgr.resolve_variant(
    "conv2d", "nvidia_sass"
  ) or mgr.resolve_variant("Conv2d", "nvidia_sass")
  assert variant is not None
  assert variant["api"] == "Macro.Conv2d"

  variant_rdna: Optional[Dict[str, Union[str, bool, int, float, list[str]]]] = mgr.resolve_variant(
    "conv2d", "rdna"
  ) or mgr.resolve_variant("Conv2d", "rdna")
  assert variant_rdna is not None
  assert variant_rdna["api"] == "; Macro.Conv2d"

  variant_add: Optional[Dict[str, Union[str, bool, int, float, list[str]]]] = mgr.resolve_variant("Add", "nvidia_sass")
  assert variant_add is not None
  assert variant_add["api"] == "FADD"


def test_nvidia_sass_roundtrip_new_macros() -> None:
  """Verifies the roundtrip compilation of the new NVIDIA_SASS macros."""
  from ml_switcheroo.core.compiler.backends.nvidia_sass import NvidiaSassBackend
  from ml_switcheroo.core.compiler.frontends.nvidia_sass.lifter import NvidiaSassLifter
  from ml_switcheroo.core.compiler.frontends.nvidia_sass.parser import NvidiaSassParser
  from ml_switcheroo.core.compiler.ir import LogicalNode

  # A dummy semantics manager that maps directly
  class DummySemantics:
    """Docstring."""

    def get_definition(self, kind: str) -> tuple[str, dict[str, str]]:
      """Gets a mock definition."""
      return (kind, {})

    def resolve_variant(self, aid: str, fw: str) -> Optional[Dict[str, str]]:
      """Resolves a mock variant."""
      if fw == "nvidia_sass":
        if aid == "Abs":
          return {"api": "FABS"}
        return {"api": f"Macro.{aid}"}
      return None

  mgr: DummySemantics = DummySemantics()
  backend: NvidiaSassBackend = NvidiaSassBackend(mgr)
  lifter: NvidiaSassLifter = NvidiaSassLifter()

  macros_to_test: list[str] = ["Conv3d", "ReLU", "Flatten", "Reshape", "Mean"]

  for kind in macros_to_test:
    g_in: LogicalGraph = LogicalGraph(nodes=[LogicalNode("n1", kind, {"k": 3, "elements": 10})])
    sass_text: str = backend.compile(g_in)

    assert f"BEGIN {kind}" in sass_text, f"Missing BEGIN comment for {kind}"

    parser: NvidiaSassParser = NvidiaSassParser(sass_text)
    ast_nodes: list[str] = parser.parse().statements  # The parser statements are typed in other files, maybe ASTNodes
    g_out: LogicalGraph = lifter.lift(ast_nodes)

    assert len(g_out.nodes) == 1, f"Failed to lift {kind} correctly"
    assert g_out.nodes[0].kind == kind, f"Lifted node kind mismatch for {kind}"

  # Test Abs (1:1 opcode)
  g_in2: LogicalGraph = LogicalGraph(nodes=[LogicalNode("n1", "Abs")])
  sass_text2: str = backend.compile(g_in2)
  assert "FABS" in sass_text2

  parser2: NvidiaSassParser = NvidiaSassParser(sass_text2)
  ast_nodes2: list[str] = parser2.parse().statements
  g_out2: LogicalGraph = lifter.lift(ast_nodes2)

  # Because it's 1:1 without a BEGIN block, it parses as an assembly instruction node
  assert len(g_out2.nodes) == 1
  assert g_out2.nodes[0].kind == "asm.FABS"


def test_rdna_sass_grounding_no_dummies() -> None:
  """Verifies that RDNA and SASS instructions are grounded and free of synthetic dummies."""
  mgr: SemanticsManager = SemanticsManager()
  all_ops = list(mgr.data.keys())

  # Ensure dummy entries are absent
  dummy_matches = [op for op in all_ops if "DUMMY_RDNA_INST" in op]
  assert not dummy_matches, f"Found synthetic dummy RDNA instructions: {dummy_matches}"

  # Verify presence of real RDNA instructions
  assert "v_add_f32_e32" in mgr.data or "global_load_dword" in mgr.data
  assert "s_branch" in mgr.data

  # Verify presence of real SASS instructions
  assert "FADD" in mgr.data or "BRA" in mgr.data
  assert "IADD3" in mgr.data
  assert "MOV" in mgr.data
  assert "FFMA" in mgr.data


def test_direct_rdna_sass_bidirectional_pivot() -> None:
  """Verifies the bidirectional translation between AMD RDNA and NVIDIA SASS assemblies via IR."""
  from ml_switcheroo.core.compiler.backends.nvidia_sass import NvidiaSassBackend
  from ml_switcheroo.core.compiler.backends.rdna import RdnaBackend
  from ml_switcheroo.core.compiler.frontends.nvidia_sass.lifter import NvidiaSassLifter
  from ml_switcheroo.core.compiler.frontends.nvidia_sass.parser import NvidiaSassParser
  from ml_switcheroo.core.compiler.frontends.rdna.lifter import RdnaLifter
  from ml_switcheroo.core.compiler.frontends.rdna.parser import RdnaParser
  from ml_switcheroo.core.compiler.ir import LogicalEdge, LogicalGraph, LogicalNode

  class MockSemantics:
    """Mock semantics manager for assembly translation."""

    def get_definition(self, kind: str) -> tuple[str, dict[str, str]]:
      """Return definition for operation kind."""
      return (kind, {})

    def resolve_variant(self, aid: str, fw: str) -> Optional[Dict[str, str]]:
      """Resolve variant for framework."""
      if fw == "nvidia_sass":
        return {"api": f"Macro.{aid}"}
      if fw == "rdna":
        return {"api": f"; Macro.{aid}"}
      return None

  mgr = MockSemantics()
  rdna_backend = RdnaBackend(mgr)
  sass_backend = NvidiaSassBackend(mgr)
  rdna_lifter = RdnaLifter()
  sass_lifter = NvidiaSassLifter()

  # 1. Start with high-level DAG: Input -> Conv2d -> ReLU -> Output
  g_initial = LogicalGraph(
    name="TestNet",
    nodes=[
      LogicalNode("x", "Input"),
      LogicalNode("c1", "Conv2d", {"k": 3, "stride": 1}),
      LogicalNode("r1", "ReLU"),
      LogicalNode("out", "Output"),
    ],
    edges=[
      LogicalEdge("x", "c1"),
      LogicalEdge("c1", "r1"),
      LogicalEdge("r1", "out"),
    ],
  )

  # 2. Compile to RDNA assembly
  rdna_asm = rdna_backend.compile(g_initial)
  assert "BEGIN Conv2d (c1)" in rdna_asm
  assert "BEGIN ReLU (r1)" in rdna_asm

  # 3. Lift RDNA assembly into neutral LogicalGraph
  rdna_ast = RdnaParser(rdna_asm).parse().statements
  g_from_rdna = rdna_lifter.lift(rdna_ast)
  assert any(n.kind == "Conv2d" for n in g_from_rdna.nodes)
  assert any(n.kind == "ReLU" for n in g_from_rdna.nodes)

  # 4. Transpile lifted graph to NVIDIA SASS assembly
  sass_asm = sass_backend.compile(g_from_rdna)
  assert "BEGIN Conv2d" in sass_asm
  assert "BEGIN ReLU" in sass_asm

  # 5. Lift NVIDIA SASS assembly into neutral LogicalGraph
  sass_ast = NvidiaSassParser(sass_asm).parse().statements
  g_from_sass = sass_lifter.lift(sass_ast)
  assert any(n.kind == "Conv2d" for n in g_from_sass.nodes)
  assert any(n.kind == "ReLU" for n in g_from_sass.nodes)

  # 6. Transpile lifted SASS graph back to AMD RDNA assembly
  rdna_asm_roundtrip = rdna_backend.compile(g_from_sass)
  assert "BEGIN Conv2d" in rdna_asm_roundtrip
  assert "BEGIN ReLU" in rdna_asm_roundtrip
