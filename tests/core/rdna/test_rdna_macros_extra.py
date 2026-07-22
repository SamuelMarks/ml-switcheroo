"""Test suite for the Rdna Macros Extra module."""

from ml_switcheroo.core.compiler.backends.rdna.macros import expand_conv2d, expand_linear
from ml_switcheroo.core.compiler.backends.rdna.synthesizer import RegisterAllocator


def test_expand_conv2d():
  """Verifies the behavior of expand conv2d."""
  alloc = RegisterAllocator()
  nodes = expand_conv2d(alloc, "conv1", {})
  assert len(nodes) > 0


def test_expand_linear():
  """Verifies the behavior of expand linear."""
  alloc = RegisterAllocator()
  nodes = expand_linear(alloc, "lin1", {})
  assert len(nodes) > 0
