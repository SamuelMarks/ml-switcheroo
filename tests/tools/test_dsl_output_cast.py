"""
Tests for ODL Schema Extension: Output Dtype Casting.
Corresponds to Limitation #12 in the Architectural roadmap.
"""

from ml_switcheroo.core.dsl import FrameworkVariant


def test_variant_output_cast_defaults_none():
  """
  Verify 'output_cast' defaults to None.
  """
  v = FrameworkVariant(api="foo")
  assert v.output_cast is None


def test_variant_output_cast_explicit():
  """
  Verify 'output_cast' stores string validation.
  """
  v = FrameworkVariant(api="argmax", output_cast="jnp.int64")
  assert v.output_cast == "jnp.int64"
