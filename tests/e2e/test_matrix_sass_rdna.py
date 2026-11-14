"""NVIDIA SASS to AMD RDNA Cross-ISA Transpilation Tests.

Verifies static disassembly parsing, lifting to LogicalGraph, and synthesis
from NVIDIA SASS assembly into AMD RDNA assembly.
"""

from pydantic import JsonValue

from ml_switcheroo import convert
from ml_switcheroo.core.compiler.backends.rdna import RdnaBackend
from ml_switcheroo.core.compiler.frontends.nvidia_sass import NvidiaSassLifter, NvidiaSassParser
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
    return {"api": f"; Macro.{abstract_id}"}


def test_sass_to_rdna_macro_cross_compilation() -> None:
  """Tests lifting a SASS macro block and synthesizing AMD RDNA."""
  sass_asm = """
// BEGIN Conv2d (conv)
FADD R1, R0, R0;
// END Conv2d (conv)
"""
  converted = convert(sass_asm, source="nvidia_sass", target="rdna")
  assert "Conv2d" in converted
  assert "BEGIN Conv2d" in converted or "v_add_f32" in converted or "v0" in converted


def test_sass_to_rdna_direct_ast_lifting() -> None:
  """Tests NvidiaSassParser + NvidiaSassLifter + RdnaBackend pipeline directly."""
  semantics = MockHardwareSemantics()
  sass_asm = """
// BEGIN Conv2d (conv)
FADD R1, R0, R0;
// END Conv2d (conv)
"""
  parser = NvidiaSassParser(sass_asm)
  statements = parser.parse().statements
  graph = NvidiaSassLifter().lift(statements)
  assert any(n.op_type == "Conv2d" for n in graph.nodes.values())

  backend = RdnaBackend(semantics)
  rdna_code = backend.compile(graph)
  assert "BEGIN Conv2d" in rdna_code
