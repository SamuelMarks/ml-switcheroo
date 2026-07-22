"""Test suite for the Dsl Output Cast module."""

from ml_switcheroo.core.dsl import FrameworkVariant


def test_variant_output_cast_defaults_none():
  """Verifies the behavior of variant output cast defaults none."""
  v = FrameworkVariant(api="foo")
  assert v.output_cast is None


def test_variant_output_cast_explicit():
  """Verifies the behavior of variant output cast explicit."""
  v = FrameworkVariant(api="argmax", output_cast="jnp.int64")
  assert v.output_cast == "jnp.int64"
