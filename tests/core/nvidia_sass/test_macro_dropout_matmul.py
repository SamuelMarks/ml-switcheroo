"""Test suite for the Dropout and MatMul NVIDIA_SASS Macros."""

import typing

from ml_switcheroo.core.compiler.backends.nvidia_sass.macros import expand_dropout, expand_linear
from ml_switcheroo.core.compiler.backends.nvidia_sass.synthesizer import RegisterAllocator
from ml_switcheroo.core.compiler.frontends.nvidia_sass.analysis import NvidiaSassAnalyzer
from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassComment, NvidiaSassInstruction, NvidiaSassNode


def test_nvidia_sass_macro_dropout() -> None:
  """Verifies that expand_dropout generates correct NVIDIA_SASS instructions."""
  allocator = RegisterAllocator()
  node_id = "drop1"
  metadata: dict[str, typing.Any] = {"p": 0.5}

  nodes: list[NvidiaSassNode] = expand_dropout(allocator, node_id, metadata)

  assert len(nodes) > 5

  comments: list[str] = [typing.cast(NvidiaSassComment, n).text for n in nodes if isinstance(n, NvidiaSassComment)]
  assert f"BEGIN Dropout ({node_id})" in comments

  opcodes: list[str] = [
    typing.cast(NvidiaSassInstruction, n).opcode for n in nodes if isinstance(n, NvidiaSassInstruction)
  ]
  assert "FSETP.GE.AND" in opcodes
  assert "FMUL" in opcodes


def test_nvidia_sass_macro_matmul() -> None:
  """Verifies that MatMul maps to linear correctly."""
  allocator = RegisterAllocator()
  node_id = "mm1"
  metadata: dict[str, typing.Any] = {"in_features": 64}

  nodes: list[NvidiaSassNode] = expand_linear(allocator, node_id, metadata)  # type: ignore
  assert len(nodes) > 5


def test_nvidia_sass_analyzer_dropout_matmul() -> None:
  """Verifies analyzer handles Dropout and MatMul."""
  instructions: list[NvidiaSassInstruction] = []
  metadata_drop: dict[str, typing.Any] = NvidiaSassAnalyzer.analyze_block("Dropout", instructions)
  assert len(metadata_drop) == 0

  metadata_mm: dict[str, typing.Any] = NvidiaSassAnalyzer.analyze_block("MatMul", instructions)
  assert len(metadata_mm) == 0
