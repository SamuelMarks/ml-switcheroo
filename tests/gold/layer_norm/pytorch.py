"""Test suite for the Pytorch module."""

import torch
import torch.nn as nn


class LayerNormModel(nn.Module):  # type: ignore[misc]
  """Test helper class LayerNormModel."""

  def __init__(self, normalized_shape: int):
    """Initializes the LayerNormModel instance.

    Args:
        normalized_shape: Test argument or fixture.
    """
    super().__init__()
    self.ln = nn.LayerNorm(normalized_shape)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Helper to forward.

    Args:
        x: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self.ln(x)
