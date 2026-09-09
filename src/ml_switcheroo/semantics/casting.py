"""Type casting utilities for semantic resolution.

Aligns numeric types and handles casting edge cases across frameworks.
"""

from typing import Dict


class TypeCaster:
  """Handles numeric type alignment between frameworks."""

  # Simplistic direct cast map
  CAST_ALIASES: Dict[str, Dict[str, str]] = {
    "pytorch": {
      "torch.float64": "float64",
      "torch.float32": "float32",
      "torch.float16": "float16",
    },
    "jax": {
      "jnp.float64": "float64",
      "jnp.float32": "float32",
      "jnp.float16": "float16",
    },
  }

  def cast(self, source_type: str, source_framework: str, target_framework: str) -> str:
    """Align numeric types between frameworks.

    Args:
        source_type: The type string from the source framework.
        source_framework: The source framework name.
        target_framework: The target framework name.

    Returns:
        The aligned type string for the target framework.
    """
    # Normalize source type
    normalized_type = source_type
    if source_framework in self.CAST_ALIASES:
      normalized_type = self.CAST_ALIASES[source_framework].get(source_type, source_type)

    # Find target type
    if target_framework in self.CAST_ALIASES:
      for tgt_type, norm_type in self.CAST_ALIASES[target_framework].items():
        if norm_type == normalized_type:
          return tgt_type

    # Fallback to normalized type if no direct map is found
    return normalized_type
