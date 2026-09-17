"""AMD RDNA to NVIDIA SASS Cross-ISA Transpilation Tests.

Verifies static disassembly parsing, lifting to LogicalGraph, and synthesis
from AMD RDNA assembly into NVIDIA SASS assembly.
"""

from typing import Dict, Optional, Tuple
from ml_switcheroo import convert
from ml_switcheroo.core.compiler.backends.nvidia_sass import NvidiaSassBackend
from ml_switcheroo.core.compiler.frontends.rdna import RdnaLifter, RdnaParser


class MockHardwareSemantics:
  """Mock semantics manager for hardware test isolation."""

  def get_definition(self, kind: str) -> Tuple[str, Dict[str, str]]:
    """Mock get_definition returning kind and metadata."""
    return (kind, {})

  def resolve_variant(self, aid: str, fw: str) -> Optional[Dict[str, str]]:
    """Mock resolve_variant returning Macro API string."""
    return {"api": f"Macro.{aid}"}


def test_rdna_to_sass_macro_cross_compilation() -> None:
  """Tests lifting an RDNA macro block and synthesizing NVIDIA SASS."""
  rdna_asm = """
; BEGIN Conv2d (conv)
v_add_f32 v1, v0, v0
; END Conv2d (conv)
"""
  converted = convert(rdna_asm, source="rdna", target="nvidia_sass")
  assert "Conv2d" in converted
  assert "BEGIN Conv2d" in converted or "FFMA" in converted or "FADD" in converted


def test_rdna_to_sass_direct_ast_lifting() -> None:
  """Tests RdnaParser + RdnaLifter + NvidiaSassBackend pipeline directly."""
  semantics = MockHardwareSemantics()
  rdna_asm = """
; BEGIN Conv2d (conv)
v_add_f32 v1, v0, v0
; END Conv2d (conv)
"""
  parser = RdnaParser(rdna_asm)
  statements = parser.parse().statements
  graph = RdnaLifter().lift(statements)
  assert any(getattr(n, "op_type", None) == "Conv2d" for n in graph.nodes.values())

  backend = NvidiaSassBackend(semantics)
  sass_code = backend.compile(graph)
  assert "BEGIN Conv2d" in sass_code
