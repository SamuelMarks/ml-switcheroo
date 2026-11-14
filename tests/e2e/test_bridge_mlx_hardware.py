"""Cross-tier Bridge Tests: Apple MLX to/from Hardware ISAs (RDNA & NVIDIA SASS).

Verifies lowering Apple MLX models to AMD RDNA and NVIDIA SASS assembly,
and lifting hardware assembly back to MLX nn.Module syntax.
"""

import ast

import pytest
from pydantic import JsonValue

from ml_switcheroo import convert
from ml_switcheroo.core.compiler.backends.nvidia_sass import NvidiaSassBackend
from ml_switcheroo.core.compiler.backends.python import PythonBackend
from ml_switcheroo.core.compiler.backends.rdna import RdnaBackend
from ml_switcheroo.core.compiler.frontends.nvidia_sass import NvidiaSassLifter, NvidiaSassParser
from ml_switcheroo.core.compiler.frontends.rdna import RdnaLifter, RdnaParser
from ml_switcheroo.core.compiler.ir import LogicalEdge, LogicalGraph, LogicalNode
from ml_switcheroo.semantics.manager import SemanticsManager


class MockBridgeSemantics(SemanticsManager):
  """Mock semantics manager for hardware bridge isolation."""

  def __init__(self) -> None:
    """Initialize empty semantics manager without disk hydration."""
    super().__init__(load_defaults=False)

  def get_definition(self, api_name: str) -> tuple[str, dict[str, JsonValue]]:
    """Return mock definition for operation kind.

    Args:
        api_name: Logical operation kind identifier.

    Returns:
        Resulting `tuple[str, dict[str, JsonValue]]` from get definition.
    """
    return (api_name, {})

  def resolve_variant(self, abstract_id: str, target_framework: str) -> dict[str, JsonValue] | None:
    """Resolve framework variant for abstract ID.

    Args:
        abstract_id: Abstract operation identifier string.
        target_framework: Framework identifier string (e.g., torch, jax, mlx).

    Returns:
        Resulting `dict[str, JsonValue] | None` from resolve variant.
    """
    if target_framework == "nvidia_sass":
      return {"api": f"Macro.{abstract_id}"}
    if target_framework == "rdna":
      return {"api": f"; Macro.{abstract_id}"}
    if target_framework == "mlx":
      if abstract_id == "Conv2d":
        return {"api": "nn.Conv2d"}
      return {"api": f"mx.{abstract_id.lower()}"}
    return None


@pytest.mark.parametrize("hw_target", ["rdna", "nvidia_sass"])
def test_mlx_to_hardware_lowering(hw_target: str) -> None:
  """Tests converting MLX source code to hardware assembly.

  Args:
      hw_target: Input `hw_target` parameter value (`str`).
  """
  mlx_code = """import mlx.core as mx
import mlx.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3)

    def __call__(self, x):
        return self.conv(x)
"""
  converted = convert(mlx_code, source="mlx", target=hw_target)
  assert "Conv2d" in converted
  if hw_target == "rdna":
    assert "v0" in converted or "; BEGIN Conv2d" in converted or ";" in converted
  else:
    assert "R" in converted or "// BEGIN Conv2d" in converted or "FADD" in converted


@pytest.mark.parametrize("hw_source", ["rdna", "nvidia_sass"])
def test_hardware_to_mlx_lifting(hw_source: str) -> None:
  """Tests lifting hardware assembly and synthesizing MLX code.

  Args:
      hw_source: Input `hw_source` parameter value (`str`).
  """
  semantics = MockBridgeSemantics()
  graph = LogicalGraph(
    name="TestNet",
    nodes={
      n.id: n
      for n in [
        LogicalNode("x", "Input"),
        LogicalNode("c1", "Conv2d", attributes={"in_channels": 3, "out_channels": 16, "kernel_size": 3}),
        LogicalNode("out", "Output"),
      ]
    },
    edges=[
      LogicalEdge("x", "c1"),
      LogicalEdge("c1", "out"),
    ],
  )

  if hw_source == "rdna":
    asm = RdnaBackend(semantics).compile(graph)
    rdna_statements = RdnaParser(asm).parse().statements
    lifted_graph = RdnaLifter().lift(rdna_statements)
  else:
    asm = NvidiaSassBackend(semantics).compile(graph)
    sass_statements = NvidiaSassParser(asm).parse().statements
    lifted_graph = NvidiaSassLifter().lift(sass_statements)

  py_code = PythonBackend(
    framework="mlx",
    semantics=semantics,
  ).compile(lifted_graph)
  parsed = ast.parse(py_code)
  assert parsed is not None
  assert "(mlx.nn.Module):" in py_code
  assert "def __call__(self, x):" in py_code
