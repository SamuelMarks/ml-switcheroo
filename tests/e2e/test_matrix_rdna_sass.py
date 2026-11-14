"""AMD RDNA to NVIDIA SASS Cross-ISA Transpilation Tests.

Verifies static disassembly parsing, lifting to LogicalGraph, and synthesis
from AMD RDNA assembly into NVIDIA SASS assembly.
"""

from pydantic import JsonValue

from ml_switcheroo import convert
from ml_switcheroo.core.compiler.backends.nvidia_sass import NvidiaSassBackend
from ml_switcheroo.core.compiler.frontends.rdna import RdnaLifter, RdnaParser
from ml_switcheroo.semantics.manager import SemanticsManager


class MockHardwareSemantics(SemanticsManager):
  """Mock semantics manager for hardware test isolation."""

  def __init__(self) -> None:
    """Initialize empty semantics manager without disk hydration."""
    super().__init__(load_defaults=False)

  def get_definition(self, api_name: str) -> tuple[str, dict[str, JsonValue]]:
    """Mock get_definition returning kind and metadata.

    Args:
        api_name: Logical operation kind identifier.

    Returns:
        Resulting `tuple[str, dict[str, JsonValue]]` from get definition.
    """
    return (api_name, {})

  def resolve_variant(self, abstract_id: str, target_framework: str) -> dict[str, JsonValue] | None:
    """Mock resolve_variant returning Macro API string.

    Args:
        abstract_id: Abstract operation identifier string.
        target_framework: Framework identifier string (e.g., torch, jax, mlx).

    Returns:
        Resulting `dict[str, JsonValue] | None` from resolve variant.
    """
    return {"api": f"Macro.{abstract_id}"}


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
  assert any(n.op_type == "Conv2d" for n in graph.nodes.values())

  backend = NvidiaSassBackend(semantics)
  sass_code = backend.compile(graph)
  assert "BEGIN Conv2d" in sass_code
