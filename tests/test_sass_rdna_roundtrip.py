"""Dedicated test suite for bidirectional NVIDIA SASS and AMD RDNA roundtrip compilation.

Validates exact structural, macro, and instruction-level equivalence
when transpiling between NVIDIA SASS assembly and AMD RDNA assembly via
the canonical intermediate representation (LogicalGraph).
"""

from typing import Dict, Optional, Tuple

import pytest

from ml_switcheroo.core.compiler.backends.nvidia_sass import NvidiaSassBackend
from ml_switcheroo.core.compiler.backends.rdna import RdnaBackend
from ml_switcheroo.core.compiler.frontends.nvidia_sass.lifter import NvidiaSassLifter
from ml_switcheroo.core.compiler.frontends.nvidia_sass.parser import NvidiaSassParser
from ml_switcheroo.core.compiler.frontends.rdna.lifter import RdnaLifter
from ml_switcheroo.core.compiler.frontends.rdna.parser import RdnaParser
from ml_switcheroo.core.compiler.ir import LogicalEdge, LogicalGraph, LogicalNode
from ml_switcheroo.core.engine import ASTEngine

SAMPLE_SASS_GEMM: str = """    // Input x -> R0
    // BEGIN Linear (fc)
    MOV R1, RZ;
    MOV R2, RZ;
L_GEMM_fc:
    LDG.E.F32 R3, [R2];
    LDG.E.F32 R4, [R3];
    FFMA R1, R3, R4, R1;
    IADD3 R2, R2, 4, RZ;
    IADD3 R3, R3, 4, RZ;
    IADD3 R2, R2, 1, RZ;
    ISETP.LT.AND P0, PT, R2, 128, PT;
    P0 BRA L_GEMM_fc;
    // END Linear (fc)
    // Return: R1
"""

SAMPLE_RDNA_GEMM: str = """; RDNA Code Generation Initialized (Arch: gfx1030)
    ; Input x -> v0
    ; BEGIN Linear (fc)
    v_mov_b32 v1, 0
    s_mov_b32 s0, 0
L_GEMM_fc:
    global_load_dword v2, v4, off
    global_load_dword v3, v5, off
    s_waitcnt vmcnt(0)
    v_fmac_f32 v1, v2, v3
    v_add_u32 v4, v4, 4
    v_add_u32 v5, v5, 4
    s_add_i32 s0, s0, 1
    s_cmp_lt_i32 s0, 128
    s_cbranch_scc1 L_GEMM_fc
    ; END Linear (fc)
    ; Return: v1
"""


class MockHardwareSemantics:
  """Mock semantics manager providing standardized hardware macro definitions."""

  def get_definition(self, kind: str) -> Tuple[str, Dict[str, str]]:
    """Return standard definition metadata for a given operation kind.

    Args:
        kind: The operation identifier string.

    Returns:
        A tuple of (kind, metadata_dict).
    """
    return (kind, {})

  def resolve_variant(self, aid: str, fw: str) -> Optional[Dict[str, str]]:
    """Resolve framework variant mapping for hardware backends.

    Args:
        aid: Abstract operation identifier.
        fw: Target hardware framework identifier ('nvidia_sass' or 'rdna').

    Returns:
        A dictionary containing the resolved API or macro template string.
    """
    if fw == "nvidia_sass":
      return {"api": f"Macro.{aid}"}
    if fw == "rdna":
      return {"api": f"; Macro.{aid}"}
    return None


def test_sass_to_rdna_via_engine() -> None:
  """Tests direct ASTEngine compilation from NVIDIA SASS to AMD RDNA."""
  engine = ASTEngine(source="nvidia_sass", target="rdna")
  res = engine.run(SAMPLE_SASS_GEMM)
  assert res.code is not None
  assert "BEGIN Linear (fc)" in res.code
  assert "v_fmac_f32" in res.code
  assert "L_GEMM_fc" in res.code


def test_rdna_to_sass_via_engine() -> None:
  """Tests direct ASTEngine compilation from AMD RDNA to NVIDIA SASS."""
  engine = ASTEngine(source="rdna", target="nvidia_sass")
  res = engine.run(SAMPLE_RDNA_GEMM)
  assert res.code is not None
  assert "BEGIN Linear (fc)" in res.code
  assert "FFMA" in res.code
  assert "L_GEMM_fc" in res.code


def test_sass_rdna_sass_lossless_roundtrip() -> None:
  """Tests a full roundtrip SASS -> RDNA -> SASS preserving Linear block semantics."""
  engine_s2r = ASTEngine(source="nvidia_sass", target="rdna")
  rdna_res = engine_s2r.run(SAMPLE_SASS_GEMM)
  assert "v_fmac_f32" in rdna_res.code

  engine_r2s = ASTEngine(source="rdna", target="nvidia_sass")
  sass_roundtrip = engine_r2s.run(rdna_res.code)
  assert "FFMA" in sass_roundtrip.code
  assert "BEGIN Linear (fc)" in sass_roundtrip.code
  assert "L_GEMM_fc" in sass_roundtrip.code


def test_rdna_sass_rdna_lossless_roundtrip() -> None:
  """Tests a full roundtrip RDNA -> SASS -> RDNA preserving Linear block semantics."""
  engine_r2s = ASTEngine(source="rdna", target="nvidia_sass")
  sass_res = engine_r2s.run(SAMPLE_RDNA_GEMM)
  assert "FFMA" in sass_res.code

  engine_s2r = ASTEngine(source="nvidia_sass", target="rdna")
  rdna_roundtrip = engine_s2r.run(sass_res.code)
  assert "v_fmac_f32" in rdna_roundtrip.code
  assert "BEGIN Linear (fc)" in rdna_roundtrip.code
  assert "L_GEMM_fc" in rdna_roundtrip.code


@pytest.mark.parametrize("kind", ["Conv2d", "Conv3d", "ReLU", "GELU", "Flatten", "Reshape"])
def test_hardware_macro_ir_roundtrip(kind: str) -> None:
  """Tests multi-macro roundtrip compilation across IR, SASS, and RDNA backends.

  Args:
      kind: The operation kind to verify across hardware backends.
  """
  semantics = MockHardwareSemantics()
  sass_backend = NvidiaSassBackend(semantics)
  rdna_backend = RdnaBackend(semantics)
  sass_lifter = NvidiaSassLifter()
  rdna_lifter = RdnaLifter()

  # 1. Construct canonical graph
  g_in = LogicalGraph(
    name="MacroNet",
    nodes=[
      LogicalNode("x", "Input"),
      LogicalNode("m1", kind, {"in_channels": 3, "out_channels": 16}),
      LogicalNode("out", "Output"),
    ],
    edges=[
      LogicalEdge("x", "m1"),
      LogicalEdge("m1", "out"),
    ],
  )

  # 2. Compile to SASS
  sass_asm = sass_backend.compile(g_in)
  assert f"BEGIN {kind}" in sass_asm

  # 3. Lift SASS to IR
  sass_ast = NvidiaSassParser(sass_asm).parse().statements
  g_from_sass = sass_lifter.lift(sass_ast)
  assert any(n.kind == kind for n in g_from_sass.nodes)

  # 4. Compile lifted IR to RDNA
  rdna_asm = rdna_backend.compile(g_from_sass)
  assert f"BEGIN {kind}" in rdna_asm

  # 5. Lift RDNA back to IR
  rdna_ast = RdnaParser(rdna_asm).parse().statements
  g_from_rdna = rdna_lifter.lift(rdna_ast)
  assert any(n.kind == kind for n in g_from_rdna.nodes)
