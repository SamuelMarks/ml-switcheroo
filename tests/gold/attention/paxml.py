"""Test suite for the Paxml module."""

import typing

from praxis.layers import attentions


class AttentionModel:
  """Test helper class AttentionModel."""

  embed_dim: typing.Any = 0
  num_heads: typing.Any = 0

  def setup(self: typing.Any) -> None:
    """Helper to setup."""
    self.create_child(
      "mha",
      attentions.DotProductAttention.HParams(num_heads=self.num_heads, dim_per_head=self.embed_dim // self.num_heads),
    )

  def __call__(self: typing.Any, query: typing.Any, key: typing.Any, value: typing.Any) -> typing.Any:
    """Executes the callable instance.

    Args:
        query: Test argument or fixture.
        key: Test argument or fixture.
        value: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self.mha(query, key, value)
