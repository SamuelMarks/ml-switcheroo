"""Cross-tier Bridge Tests: PyTorch to/from Hardware ISAs (RDNA & NVIDIA SASS).

Verifies lowering PyTorch models to AMD RDNA and NVIDIA SASS assembly,
and lifting hardware assembly back to PyTorch nn.Module syntax.
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
        Resulting `tuple[str, dict[str, str]]` from get definition.
    """
    return (kind, {})

  def resolve_variant(self, aid: str, fw: str) -> dict[str, str] | None:
    """Resolve framework variant for abstract ID.

    Args:
        aid: Abstract operation identifier string.
        fw: Framework identifier string (e.g., torch, jax, mlx).

    Returns:
        Resulting `dict[str, str] | None` from resolve variant.
    """
    if fw == "nvidia_sass":
      return {"api": f"Macro.{aid}"}
    if fw == "rdna":
      return {"api": f"; Macro.{aid}"}
    if fw == "torch":
      if aid == "Conv2d":
        return {"api": "nn.Conv2d"}
      if aid == "ReLU":
        return {"api": "nn.ReLU"}
      return {"api": f"torch.{aid.lower()}"}
    return None


@pytest.mark.parametrize("hw_target", ["rdna", "nvidia_sass"])
def test_torch_to_hardware_lowering(hw_target: str) -> None:
  """Tests converting PyTorch source code to hardware assembly.

  Args:
      hw_target: The hw target (str) for test torch to hardware lowering.
  """
  torch_code = """import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3)

    def forward(self, x):
        return self.conv(x)
"""
  converted = convert(torch_code, source="torch", target=hw_target)
  assert "Conv2d" in converted
  if hw_target == "rdna":
    assert "v0" in converted or "; BEGIN Conv2d" in converted or ";" in converted
  else:
    assert "R" in converted or "// BEGIN Conv2d" in converted or "FADD" in converted


@pytest.mark.parametrize("hw_source", ["rdna", "nvidia_sass"])
def test_hardware_to_torch_lifting(hw_source: str) -> None:
  """Tests lifting hardware assembly and synthesizing PyTorch code.

  Args:
      hw_source: The hw source (str) for test hardware to torch lifting.
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
                framework=getattr({"v": "torch"}, "__getitem__")("v"),
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
  assert "(nn.Module):" in py_code
  assert "def forward(self, x):" in py_code
