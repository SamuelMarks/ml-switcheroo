"""Test suite for the final set of generic NVIDIA_SASS Macros."""

from ml_switcheroo.core.compiler.backends.nvidia_sass.macros import (
  expand_generic_activation,
  expand_generic_dropout,
  expand_generic_linalg,
  expand_generic_loss,
  expand_generic_norm,
  expand_generic_reduction,
)
from ml_switcheroo.core.compiler.backends.nvidia_sass.synthesizer import RegisterAllocator
from ml_switcheroo.core.compiler.frontends.nvidia_sass.analysis import NvidiaSassAnalyzer
from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassInstruction


def test_nvidia_sass_macro_final() -> None:
  """Verifies generic macros."""
  allocator = RegisterAllocator()
  assert len(expand_generic_norm(allocator, "n1", {})) > 2  # type: ignore
  assert len(expand_generic_activation(allocator, "a1", {})) > 2  # type: ignore
  assert len(expand_generic_linalg(allocator, "l1", {})) > 2  # type: ignore
  assert len(expand_generic_reduction(allocator, "r1", {})) > 2  # type: ignore
  assert len(expand_generic_loss(allocator, "ls1", {})) > 2  # type: ignore
  assert len(expand_generic_dropout(allocator, "d1", {})) > 2  # type: ignore


def test_nvidia_sass_analyzer_final() -> None:
  """Verifies analyzer."""
  instructions: list[NvidiaSassInstruction] = []
  assert len(NvidiaSassAnalyzer.analyze_block("BatchNorm1d", instructions)) == 0
  assert len(NvidiaSassAnalyzer.analyze_block("Softmax", instructions)) == 0
  assert len(NvidiaSassAnalyzer.analyze_block("BMM", instructions)) == 0
  assert len(NvidiaSassAnalyzer.analyze_block("Sum", instructions)) == 0
  assert len(NvidiaSassAnalyzer.analyze_block("BCEWithLogitsLoss", instructions)) == 0
  assert len(NvidiaSassAnalyzer.analyze_block("Dropout2d", instructions)) == 0
