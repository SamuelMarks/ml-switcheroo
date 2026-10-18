"""NVIDIA SASS to AMD RDNA Cross-ISA Transpilation Tests.

Verifies static disassembly parsing, lifting to LogicalGraph, and synthesis
from NVIDIA SASS assembly into AMD RDNA assembly.
"""

import typing

from typing import Dict, Optional, Tuple

from ml_switcheroo import convert
from ml_switcheroo.core.compiler.backends.rdna import RdnaBackend
from ml_switcheroo.core.compiler.frontends.nvidia_sass import NvidiaSassLifter, NvidiaSassParser


class MockHardwareSemantics:
  """Mock semantics manager for hardware test isolation."""

  def get_definition(self, kind: str) -> Tuple[str, Dict[str, str]]:
    """Mock get_definition returning kind and metadata.

    Returns:
        Computed test result or fixture value.
    """
    return (kind, {})

  def resolve_variant(self, aid: str, fw: str) -> Optional[Dict[str, str]]:
    """Mock resolve_variant returning Macro API string.

    Args:
        aid: Test argument or fixture.
        fw: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return {"api": f"; Macro.{aid}"}


def test_sass_to_rdna_macro_cross_compilation() -> None:
  """Tests lifting a SASS macro block and synthesizing AMD RDNA."""
  sass_asm: typing.Any = """
// BEGIN Conv2d (conv)
FADD R1, R0, R0;
// END Conv2d (conv)
"""
  converted: typing.Any = convert(sass_asm, source="nvidia_sass", target="rdna")
  assert "Conv2d" in converted
  assert "BEGIN Conv2d" in converted or "v_add_f32" in converted or "v0" in converted


def test_sass_to_rdna_direct_ast_lifting() -> None:
  """Tests NvidiaSassParser + NvidiaSassLifter + RdnaBackend pipeline directly."""
  semantics: typing.Any = MockHardwareSemantics()
  sass_asm: typing.Any = """
// BEGIN Conv2d (conv)
FADD R1, R0, R0;
// END Conv2d (conv)
"""
  parser: typing.Any = NvidiaSassParser(sass_asm)
  statements: typing.Any = parser.parse().statements
  graph: typing.Any = NvidiaSassLifter().lift(statements)
  assert any(getattr(n, "op_type", None) == "Conv2d" for n in graph.nodes.values())

  backend: typing.Any = RdnaBackend(semantics)
  rdna_code: typing.Any = backend.compile(graph)
  assert "BEGIN Conv2d" in rdna_code
