"""Test suite for Conv and Pooling SASS Macros."""

from ml_switcheroo.core.compiler.backends.sass.macros import (
  expand_conv1d,
  expand_depthwiseconv2d,
  expand_convtranspose,
  expand_pool1d,
  expand_pool3d,
  expand_adaptivepool,
)
from ml_switcheroo.core.compiler.frontends.sass.analysis import SassAnalyzer
from ml_switcheroo.core.compiler.backends.sass.synthesizer import RegisterAllocator


def test_sass_macro_convs() -> None:
  """Verifies conv macros."""
  allocator = RegisterAllocator()
  assert len(expand_conv1d(allocator, "c1", {"k": 3})) > 5
  assert len(expand_depthwiseconv2d(allocator, "c2", {"k": 3})) > 5
  assert len(expand_convtranspose(allocator, "c3", {})) > 2


def test_sass_macro_pools() -> None:
  """Verifies pool macros."""
  allocator = RegisterAllocator()
  assert len(expand_pool1d(allocator, "p1", {})) > 2
  assert len(expand_pool3d(allocator, "p3", {})) > 2
  assert len(expand_adaptivepool(allocator, "pa", {})) > 2


def test_sass_analyzer_conv_pool():
  """Verifies analyzer."""
  instructions = []
  assert len(SassAnalyzer.analyze_block("Conv1d", instructions)) == 0
  assert len(SassAnalyzer.analyze_block("AvgPool1d", instructions)) == 0
