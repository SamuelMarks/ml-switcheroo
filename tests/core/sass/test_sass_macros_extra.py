"""Test module."""

from ml_switcheroo.core.compiler.backends.sass.macros import expand_conv2d, expand_linear
from ml_switcheroo.core.compiler.backends.sass.synthesizer import RegisterAllocator


def test_expand_conv2d():
  """Test function."""
  alloc = RegisterAllocator()
  nodes = expand_conv2d(alloc, "conv1", {})
  assert len(nodes) > 0


def test_expand_linear():
  """Test function."""
  alloc = RegisterAllocator()
  nodes = expand_linear(alloc, "lin1", {})
  assert len(nodes) > 0
