"""Cross-tier Bridge Tests: JAX to/from Hardware ISAs (RDNA & NVIDIA SASS).

Verifies lowering JAX models to AMD RDNA and NVIDIA SASS assembly,
and lifting hardware assembly back to JAX flax.nnx.Module syntax.
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
    """Initialize mock semantics manager without disk I/O."""

  def get_definition(self, name: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Return mock definition for operation kind.

    Args:
        name: Operation name to look up.

    Returns:
        Tuple of abstract operation ID and empty metadata dict.
    """
    return (name, {})

  def resolve_variant(self, abstract_id: str, target_fw: str) -> dict[str, JsonValue] | None:
    """Resolve framework variant for abstract ID.

    Args:
        abstract_id: Abstract operation identifier.
        target_fw: Target framework identifier.

    Returns:
        Variant mapping dictionary or None.
    """
    if target_fw == "nvidia_sass":
      return {"api": f"Macro.{abstract_id}"}
    if target_fw == "rdna":
      return {"api": f"; Macro.{abstract_id}"}
    if target_fw == "jax":
      if abstract_id == "Conv2d":
        return {"api": "flax.nnx.Conv"}
      return {"api": f"jnp.{abstract_id.lower()}"}
    return None


@pytest.mark.parametrize("hw_target", ["rdna", "nvidia_sass"])
def test_jax_to_hardware_lowering(hw_target: str) -> None:
  """Tests converting JAX source code to hardware assembly.

  Args:
      hw_target: The hw target (str) for test jax to hardware lowering.
  """
  jax_code = """import jax.numpy as jnp
import flax.nnx as nnx

class Net(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.conv = nnx.Conv(3, 16, 3, rngs=rngs)

    def __call__(self, x):
        return self.conv(x)
"""
  converted = convert(jax_code, source="jax", target=hw_target)
  assert "conv" in converted
  if hw_target == "rdna":
    assert "v0" in converted or "; BEGIN Conv" in converted or ";" in converted
  else:
    assert "R" in converted or "// BEGIN Conv" in converted or "FADD" in converted


@pytest.mark.parametrize("hw_source", ["rdna", "nvidia_sass"])
def test_hardware_to_jax_lifting(hw_source: str) -> None:
  """Tests lifting hardware assembly and synthesizing JAX code.

  Args:
      hw_source: The hw source (str) for test hardware to jax lifting.
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
    framework="jax",
    semantics=semantics,
  ).compile(lifted_graph)
  parsed = ast.parse(py_code)
  assert any(isinstance(node, ast.ClassDef) for node in parsed.body)
  assert "def __call__(self, x):" in py_code
  assert "nnx.Rngs" in py_code
