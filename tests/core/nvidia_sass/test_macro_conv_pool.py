"""Test suite for Conv and Pooling NVIDIA_SASS Macros."""

from ml_switcheroo.core.compiler.backends.nvidia_sass.macros import (
  expand_adaptivepool,
  expand_conv1d,
  expand_convtranspose,
  expand_depthwiseconv2d,
  expand_pool1d,
  expand_pool3d,
)
from ml_switcheroo.core.compiler.backends.nvidia_sass.synthesizer import RegisterAllocator
from ml_switcheroo.core.compiler.frontends.nvidia_sass.analysis import NvidiaSassAnalyzer
from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassInstruction


def test_nvidia_sass_macro_convs() -> None:
  """Verifies conv macros."""
  allocator = RegisterAllocator()
  assert len(expand_conv1d(allocator, "c1", {"k": 3})) > 5  # type: ignore
  assert len(expand_depthwiseconv2d(allocator, "c2", {"k": 3})) > 5  # type: ignore
  assert len(expand_convtranspose(allocator, "c3", {})) > 2


def test_nvidia_sass_macro_pools() -> None:
  """Verifies pool macros."""
  allocator = RegisterAllocator()
  assert len(expand_pool1d(allocator, "p1", {})) > 2  # type: ignore
  assert len(expand_pool3d(allocator, "p3", {})) > 2  # type: ignore
  assert len(expand_adaptivepool(allocator, "pa", {})) > 2  # type: ignore


def test_nvidia_sass_analyzer_conv_pool() -> None:
  """Verifies analyzer."""
  instructions: list[NvidiaSassInstruction] = []
  assert len(NvidiaSassAnalyzer.analyze_block("Conv1d", instructions)) == 0
  assert len(NvidiaSassAnalyzer.analyze_block("AvgPool1d", instructions)) == 0
