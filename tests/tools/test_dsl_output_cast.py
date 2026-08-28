"""Test suite for the Dsl Output Cast module."""

from ml_switcheroo.core.dsl import FrameworkVariant


def test_variant_output_cast_defaults_none() -> None:
  """Verifies the behavior of variant output cast defaults none."""
  v: FrameworkVariant = FrameworkVariant(api="foo")
  assert getattr(v, "output_cast") is None


def test_variant_output_cast_explicit() -> None:
  """Verifies the behavior of variant output cast explicit."""
  v: FrameworkVariant = FrameworkVariant(api="argmax", output_cast="jnp.int64")
  assert getattr(v, "output_cast") == "jnp.int64"
