"""Test suite for the Activation NVIDIA_SASS Macros."""

import typing

from ml_switcheroo.core.compiler.backends.nvidia_sass.macros import expand_gelu, expand_sigmoid, expand_tanh
from ml_switcheroo.core.compiler.backends.nvidia_sass.synthesizer import RegisterAllocator
from ml_switcheroo.core.compiler.frontends.nvidia_sass.analysis import NvidiaSassAnalyzer
from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassComment, NvidiaSassInstruction, NvidiaSassNode


def test_nvidia_sass_macro_sigmoid() -> None:
  """Verifies that expand_sigmoid generates correct NVIDIA_SASS instructions."""
  allocator = RegisterAllocator()
  node_id = "sig1"
  metadata: dict[str, typing.Any] = {}

  nodes: list[NvidiaSassNode] = expand_sigmoid(allocator, node_id, metadata)
  assert len(nodes) > 5

  comments: list[str] = [typing.cast(NvidiaSassComment, n).text for n in nodes if isinstance(n, NvidiaSassComment)]
  assert f"BEGIN Sigmoid ({node_id})" in comments

  opcodes: list[str] = [
    typing.cast(NvidiaSassInstruction, n).opcode for n in nodes if isinstance(n, NvidiaSassInstruction)
  ]
  assert "MUFU" in opcodes
  assert "FADD" in opcodes


def test_nvidia_sass_macro_tanh() -> None:
  """Verifies that expand_tanh generates correct NVIDIA_SASS instructions."""
  allocator = RegisterAllocator()
  node_id = "tanh1"
  metadata: dict[str, typing.Any] = {}

  nodes: list[NvidiaSassNode] = expand_tanh(allocator, node_id, metadata)
  assert len(nodes) > 2

  comments: list[str] = [typing.cast(NvidiaSassComment, n).text for n in nodes if isinstance(n, NvidiaSassComment)]
  assert f"BEGIN Tanh ({node_id})" in comments


def test_nvidia_sass_macro_gelu() -> None:
  """Verifies that expand_gelu generates correct NVIDIA_SASS instructions."""
  allocator = RegisterAllocator()
  node_id = "gelu1"
  metadata: dict[str, typing.Any] = {}

  nodes: list[NvidiaSassNode] = expand_gelu(allocator, node_id, metadata)
  assert len(nodes) > 5

  comments: list[str] = [typing.cast(NvidiaSassComment, n).text for n in nodes if isinstance(n, NvidiaSassComment)]
  assert f"BEGIN GELU ({node_id})" in comments

  opcodes: list[str] = [
    typing.cast(NvidiaSassInstruction, n).opcode for n in nodes if isinstance(n, NvidiaSassInstruction)
  ]
  assert "MUFU" in opcodes
  assert "FMUL" in opcodes


def test_nvidia_sass_analyzer_activations() -> None:
  """Verifies analyzer handles activations."""
  instructions: list[NvidiaSassInstruction] = []
  assert len(NvidiaSassAnalyzer.analyze_block("Sigmoid", instructions)) == 0
  assert len(NvidiaSassAnalyzer.analyze_block("Tanh", instructions)) == 0
  assert len(NvidiaSassAnalyzer.analyze_block("GELU", instructions)) == 0
