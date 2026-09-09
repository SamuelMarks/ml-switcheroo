"""Test suite for the Attention NVIDIA_SASS Macros."""

import typing

from ml_switcheroo.core.compiler.backends.nvidia_sass.macros import (
  expand_multiheadattention,
  expand_transformer,
  expand_transformerdecoder,
  expand_transformerencoder,
)
from ml_switcheroo.core.compiler.backends.nvidia_sass.synthesizer import RegisterAllocator
from ml_switcheroo.core.compiler.frontends.nvidia_sass.analysis import NvidiaSassAnalyzer
from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassComment, NvidiaSassInstruction, NvidiaSassNode


def test_nvidia_sass_macro_multiheadattention() -> None:
  """Verifies that expand_multiheadattention generates correct NVIDIA_SASS instructions."""
  allocator = RegisterAllocator()
  node_id = "mha1"
  metadata: dict[str, typing.Any] = {}

  nodes: list[NvidiaSassNode] = expand_multiheadattention(allocator, node_id, metadata)
  assert len(nodes) > 5

  comments: list[str] = [typing.cast(NvidiaSassComment, n).text for n in nodes if isinstance(n, NvidiaSassComment)]
  assert f"BEGIN MultiheadAttention ({node_id})" in comments


def test_nvidia_sass_macro_transformer() -> None:
  """Verifies that expand_transformer generates correct NVIDIA_SASS instructions."""
  allocator = RegisterAllocator()
  node_id = "tf1"
  metadata: dict[str, typing.Any] = {}

  nodes: list[NvidiaSassNode] = expand_transformer(allocator, node_id, metadata)
  assert len(nodes) > 3

  comments: list[str] = [typing.cast(NvidiaSassComment, n).text for n in nodes if isinstance(n, NvidiaSassComment)]
  assert f"BEGIN Transformer ({node_id})" in comments


def test_nvidia_sass_macro_transformer_enc_dec() -> None:
  """Verifies enc/dec macros."""
  allocator = RegisterAllocator()
  node_id = "enc1"
  metadata: dict[str, typing.Any] = {}

  nodes: list[NvidiaSassNode] = expand_transformerencoder(allocator, node_id, metadata)
  assert len(nodes) >= 2

  nodes2: list[NvidiaSassNode] = expand_transformerdecoder(allocator, node_id, metadata)
  assert len(nodes2) >= 2


def test_nvidia_sass_analyzer_attention() -> None:
  """Verifies analyzer handles attention ops safely."""
  instructions: list[NvidiaSassInstruction] = []
  assert len(NvidiaSassAnalyzer.analyze_block("MultiheadAttention", instructions)) == 0
  assert len(NvidiaSassAnalyzer.analyze_block("Transformer", instructions)) == 0
