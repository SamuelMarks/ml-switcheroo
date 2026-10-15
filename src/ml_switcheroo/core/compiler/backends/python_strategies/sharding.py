"""Sharding formatting strategies for distributed tensor compilation."""

from typing import Any


def format_partition_spec(sharding: Any) -> str:
  """Format a sharding annotation into a JAX `PartitionSpec` constructor string.

  Args:
      sharding: Sharding metadata object containing an `axes` sequence.

  Returns:
      String representation of `jax.sharding.PartitionSpec(...)`.
  """
  axes = []
  for axis in sharding.axes:
    if axis is None:
      axes.append("None")
    elif isinstance(axis, str):
      axes.append(f"'{axis}'")
    elif isinstance(axis, tuple):
      t_str = ", ".join(f"'{a}'" for a in axis)
      axes.append(f"({t_str})")
  return f"jax.sharding.PartitionSpec({', '.join(axes)})"


def format_partition_spec_tf(sharding: Any) -> str:
  """Format a sharding annotation into a TensorFlow/Keras layout list string.

  Args:
      sharding: Sharding metadata object containing an `axes` sequence.

  Returns:
      Bracketed list string of TensorFlow/Keras axis placement names.
  """
  placements = []
  for axis in sharding.axes:
    if axis is None:
      placements.append("None")
    elif isinstance(axis, str):
      placements.append(f"'{axis}'")
    else:
      placements.append("'*'")
  return f"[{', '.join(placements)}]"


def format_partition_spec_torch(sharding: Any) -> str:
  """Format a sharding annotation into PyTorch DTensor placement constructors.

  Args:
      sharding: Sharding metadata object containing an `axes` sequence.

  Returns:
      Comma-separated string of `Replicate()` and `Shard(0)` placements.
  """
  placements = []
  for axis in sharding.axes:
    if axis is None:
      placements.append("Replicate()")
    else:
      placements.append("Shard(0)")
  return ", ".join(placements)
