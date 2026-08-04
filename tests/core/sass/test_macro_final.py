"""Test suite for the final set of generic SASS Macros."""

from ml_switcheroo.core.compiler.backends.sass.macros import (
  expand_generic_norm,
  expand_generic_activation,
  expand_generic_linalg,
  expand_generic_reduction,
  expand_generic_loss,
  expand_generic_dropout,
)
from ml_switcheroo.core.compiler.frontends.sass.analysis import SassAnalyzer
from ml_switcheroo.core.compiler.backends.sass.synthesizer import RegisterAllocator


def test_sass_macro_final() -> None:
  """Verifies generic macros."""
  allocator = RegisterAllocator()
  assert len(expand_generic_norm(allocator, "n1", {})) > 2
  assert len(expand_generic_activation(allocator, "a1", {})) > 2
  assert len(expand_generic_linalg(allocator, "l1", {})) > 2
  assert len(expand_generic_reduction(allocator, "r1", {})) > 2
  assert len(expand_generic_loss(allocator, "ls1", {})) > 2
  assert len(expand_generic_dropout(allocator, "d1", {})) > 2


def test_sass_analyzer_final():
  """Verifies analyzer."""
  instructions = []
  assert len(SassAnalyzer.analyze_block("BatchNorm1d", instructions)) == 0
  assert len(SassAnalyzer.analyze_block("Softmax", instructions)) == 0
  assert len(SassAnalyzer.analyze_block("BMM", instructions)) == 0
  assert len(SassAnalyzer.analyze_block("Sum", instructions)) == 0
  assert len(SassAnalyzer.analyze_block("BCEWithLogitsLoss", instructions)) == 0
  assert len(SassAnalyzer.analyze_block("Dropout2d", instructions)) == 0
