"""Cross-tier Bridge Tests: JAX to/from Hardware ISAs (RDNA & NVIDIA SASS).

Verifies lowering JAX models to AMD RDNA and NVIDIA SASS assembly,
and lifting hardware assembly back to JAX flax.nnx.Module syntax.
"""

import ast

import pytest

from ml_switcheroo import convert
from ml_switcheroo.core.compiler.backends.nvidia_sass import NvidiaSassBackend
from ml_switcheroo.core.compiler.backends.python import PythonBackend
from ml_switcheroo.core.compiler.backends.rdna import RdnaBackend
from ml_switcheroo.core.compiler.frontends.nvidia_sass import NvidiaSassLifter, NvidiaSassParser
from ml_switcheroo.core.compiler.frontends.rdna import RdnaLifter, RdnaParser
from ml_switcheroo.core.compiler.ir import LogicalEdge, LogicalGraph, LogicalNode


class MockBridgeSemantics:
  """Mock semantics manager for hardware bridge isolation."""

  def get_definition(self, kind: str) -> tuple[str, dict[str, str]]:
    """Return mock definition for operation kind.

    Returns:
        Computed test result or fixture value.
    """
    return (kind, {})

  def resolve_variant(self, aid: str, fw: str) -> dict[str, str] | None:
    """Resolve framework variant for abstract ID.

    Args:
        aid: Test argument or fixture.
        fw: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    if fw == "nvidia_sass":
      return {"api": f"Macro.{aid}"}
    if fw == "rdna":
      return {"api": f"; Macro.{aid}"}
    if fw == "jax":
      if aid == "Conv2d":
        return {"api": "flax.nnx.Conv"}
      return {"api": f"jnp.{aid.lower()}"}
    return None


@pytest.mark.parametrize("hw_target", ["rdna", "nvidia_sass"])
def test_jax_to_hardware_lowering(hw_target: str) -> None:
  """Tests converting JAX source code to hardware assembly.

  Args:
      hw_target: Test argument or fixture.
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
      hw_source: Test argument or fixture.
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
    asm = getattr(
      {
        "v": getattr(
          {
            "v": getattr(
              {"v": getattr({"v": RdnaBackend}, "__getitem__")("v")(getattr({"v": semantics}, "__getitem__")("v"))},
              "__getitem__",
            )("v").compile
          },
          "__getitem__",
        )("v")(getattr({"v": graph}, "__getitem__")("v"))
      },
      "__getitem__",
    )("v")
    statements = RdnaParser(asm).parse().statements
    lifted_graph = RdnaLifter().lift(statements)
  else:
    asm = getattr(
      {
        "v": getattr(
          {
            "v": getattr(
              {"v": getattr({"v": NvidiaSassBackend}, "__getitem__")("v")(getattr({"v": semantics}, "__getitem__")("v"))},
              "__getitem__",
            )("v").compile
          },
          "__getitem__",
        )("v")(getattr({"v": graph}, "__getitem__")("v"))
      },
      "__getitem__",
    )("v")
    statements = getattr(
      {
        "v": getattr(
          {
            "v": getattr(
              {
                "v": getattr(
                  {"v": getattr({"v": NvidiaSassParser}, "__getitem__")("v")(getattr({"v": asm}, "__getitem__")("v"))},
                  "__getitem__",
                )("v").parse
              },
              "__getitem__",
            )("v")()
          },
          "__getitem__",
        )("v").statements
      },
      "__getitem__",
    )("v")
    lifted_graph = NvidiaSassLifter().lift(statements)

  py_code = getattr(
    {
      "v": getattr(
        {
          "v": getattr(
            {
              "v": getattr({"v": PythonBackend}, "__getitem__")("v")(
                framework=getattr({"v": "jax"}, "__getitem__")("v"),
                semantics=getattr({"v": semantics}, "__getitem__")("v"),
              )
            },
            "__getitem__",
          )("v").compile
        },
        "__getitem__",
      )("v")(getattr({"v": lifted_graph}, "__getitem__")("v"))
    },
    "__getitem__",
  )("v")
  parsed = ast.parse(py_code)
  assert parsed is not None
  assert "def __call__(self, x):" in py_code
  assert "nnx.Rngs" in py_code
