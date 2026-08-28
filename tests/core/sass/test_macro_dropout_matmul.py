"""Test suite for the Dropout and MatMul SASS Macros."""

import typing
from ml_switcheroo.core.compiler.backends.sass.macros import expand_dropout, expand_linear
from ml_switcheroo.core.compiler.frontends.sass.analysis import SassAnalyzer
from ml_switcheroo.core.compiler.backends.sass.synthesizer import RegisterAllocator
from ml_switcheroo.core.compiler.frontends.sass.cst import SassInstruction, SassComment, SassNode


def test_sass_macro_dropout() -> None:
  """Verifies that expand_dropout generates correct SASS instructions."""
  allocator = RegisterAllocator()
  node_id = "drop1"
  metadata: dict[str, typing.Any] = {"p": 0.5}

  nodes: list[SassNode] = expand_dropout(allocator, node_id, metadata)

  assert len(nodes) > 5

  comments: list[str] = [typing.cast(SassComment, n).text for n in nodes if isinstance(n, SassComment)]
  assert f"BEGIN Dropout ({node_id})" in comments

  opcodes: list[str] = [typing.cast(SassInstruction, n).opcode for n in nodes if isinstance(n, SassInstruction)]
  assert "FSETP.GE.AND" in opcodes
  assert "FMUL" in opcodes


def test_sass_macro_matmul() -> None:
  """Verifies that MatMul maps to linear correctly."""
  allocator = RegisterAllocator()
  node_id = "mm1"
  metadata: dict[str, typing.Any] = {"in_features": 64}

  nodes: list[SassNode] = expand_linear(allocator, node_id, metadata)  # type: ignore
  assert len(nodes) > 5


def test_sass_analyzer_dropout_matmul() -> None:
  """Verifies analyzer handles Dropout and MatMul."""
  instructions: list[SassInstruction] = []
  metadata_drop: dict[str, typing.Any] = SassAnalyzer.analyze_block("Dropout", instructions)
  assert len(metadata_drop) == 0

  metadata_mm: dict[str, typing.Any] = SassAnalyzer.analyze_block("MatMul", instructions)
  assert len(metadata_mm) == 0
