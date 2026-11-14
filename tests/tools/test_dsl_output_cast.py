"""Test suite for the Dsl Output Cast module."""

from ml_switcheroo.core.dsl import FrameworkVariant


def test_variant_output_cast_defaults_none() -> None:
  """Verify `FrameworkVariant` handles variant output cast defaults none as expected."""
  v: FrameworkVariant = FrameworkVariant(api="foo")
  assert v.output_cast is None


def test_variant_output_cast_explicit() -> None:
  """Verify `FrameworkVariant` handles variant output cast explicit as expected."""
  v: FrameworkVariant = FrameworkVariant(api="argmax", output_cast="jnp.int64")
  assert v.output_cast == "jnp.int64"
