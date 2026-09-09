"""Test suite for the Loss NVIDIA_SASS Macros."""

import typing

from ml_switcheroo.core.compiler.backends.nvidia_sass.macros import expand_crossentropyloss, expand_mseloss
from ml_switcheroo.core.compiler.backends.nvidia_sass.synthesizer import RegisterAllocator
from ml_switcheroo.core.compiler.frontends.nvidia_sass.analysis import NvidiaSassAnalyzer
from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import (
  NvidiaSassComment,
  NvidiaSassImmediate,
  NvidiaSassInstruction,
  NvidiaSassNode,
  NvidiaSassRegister,
)


def test_nvidia_sass_macro_mseloss() -> None:
  """Verifies that expand_mseloss generates correct NVIDIA_SASS instructions."""
  allocator = RegisterAllocator()
  node_id = "mse1"
  metadata: dict[str, typing.Any] = {"elements": 32, "reduction": "mean"}

  nodes: list[NvidiaSassNode] = expand_mseloss(allocator, node_id, metadata)
  assert len(nodes) > 10

  comments: list[str] = [typing.cast(NvidiaSassComment, n).text for n in nodes if isinstance(n, NvidiaSassComment)]
  assert f"BEGIN MSELoss ({node_id})" in comments

  opcodes: list[str] = [
    typing.cast(NvidiaSassInstruction, n).opcode for n in nodes if isinstance(n, NvidiaSassInstruction)
  ]
  assert "FMUL" in opcodes
  assert "FADD" in opcodes


def test_nvidia_sass_macro_crossentropyloss() -> None:
  """Verifies that expand_crossentropyloss generates correct NVIDIA_SASS instructions."""
  allocator = RegisterAllocator()
  node_id = "ce1"
  metadata: dict[str, typing.Any] = {"elements": 64}

  nodes: list[NvidiaSassNode] = expand_crossentropyloss(allocator, node_id, metadata)
  assert len(nodes) > 10

  comments: list[str] = [typing.cast(NvidiaSassComment, n).text for n in nodes if isinstance(n, NvidiaSassComment)]
  assert f"BEGIN CrossEntropyLoss ({node_id})" in comments

  opcodes: list[str] = [
    typing.cast(NvidiaSassInstruction, n).opcode for n in nodes if isinstance(n, NvidiaSassInstruction)
  ]
  assert "MUFU" in opcodes
  assert "FMUL" in opcodes


def test_nvidia_sass_analyzer_loss() -> None:
  """Verifies analyzer handles loss limits."""
  instructions = [
    NvidiaSassInstruction(
      opcode="ISETP.LT.AND",
      operands=[
        NvidiaSassRegister(name="P0"),
        NvidiaSassRegister(name="PT"),
        NvidiaSassRegister(name="R0"),
        NvidiaSassImmediate(value=100),
        NvidiaSassRegister(name="PT"),
      ],
    )
  ]
  metadata_mse: dict[str, typing.Any] = NvidiaSassAnalyzer.analyze_block("MSELoss", instructions)
  assert metadata_mse["elements"] == 100

  metadata_ce: dict[str, typing.Any] = NvidiaSassAnalyzer.analyze_block("CrossEntropyLoss", instructions)
  assert metadata_ce["elements"] == 100
