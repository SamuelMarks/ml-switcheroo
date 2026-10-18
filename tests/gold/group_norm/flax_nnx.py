"""Test suite for the Flax Nnx module."""

import typing

from flax import nnx


class GNModel(nnx.Module):
  """Test helper class GNModel."""

  def __init__(self, num_groups: typing.Any, num_channels: typing.Any, rngs: typing.Any):
    """Initializes the GNModel instance.

    Args:
        num_groups: Test argument or fixture.
        num_channels: Test argument or fixture.
        rngs: Test argument or fixture.
    """
    self.gn = getattr(nnx, "GroupNorm")(num_groups=num_groups, rngs=rngs)

  def __call__(self, x: typing.Any) -> typing.Any:
    """Executes the callable instance.

    Args:
        x: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self.gn(x)
