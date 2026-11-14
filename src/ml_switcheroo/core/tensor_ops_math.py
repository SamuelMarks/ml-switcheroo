"""Differentiable Scalar, Reduction, Shape, and Elementwise Math Kernels for On-Device Tracing.

Provides pure-Python numerical and VJP helper functions delegated to by
``OnDeviceTraceTensor`` in ``ml_switcheroo.core.tensor_core``.
"""

from __future__ import annotations

import math
import random
import typing

if typing.TYPE_CHECKING:
  from ml_switcheroo.core.tensor_core import OnDeviceTraceTensor


def _prod(shape: typing.Sequence[int]) -> int:
  """Compute the product of dimensions in a shape sequence.

  Args:
      shape: Sequence of integer dimension sizes.

  Returns:
      Total number of elements (minimum 1 for scalar ``()``).
  """
  total = 1
  for d in shape:
    total *= max(1, int(d))
  return total


def _generate_seeded_values(count: int, seed: int, dtype: str, scale: float = 0.5) -> list[float]:
  """Generate deterministic pseudo-random tensor values for a given seed.

  Args:
      count: Number of scalar elements to generate.
      seed: Deterministic integer seed.
      dtype: Target dtype string (``'int32'``, ``'float32'``, ``'float16'``).
      scale: Magnitude scale factor for floating-point values.

  Returns:
      Flat list of deterministic float values.
  """
  rng = random.Random(seed)
  if "int" in dtype:
    return [float(rng.randint(0, 7)) for _ in range(count)]
  vals: list[float] = []
  for i in range(count):
    raw = rng.uniform(-scale, scale)
    if abs(raw) < 0.05:
      raw = 0.15 if (i % 2 == 0) else -0.15
    vals.append(round(raw, 6))
  return vals


def _l2_norm(values: typing.Sequence[float]) -> float:
  """Compute the Euclidean (L2) norm of a sequence of floats.

  Args:
      values: Sequence of scalar float values.

  Returns:
      L2 norm rounded to 6 decimal places.
  """
  return round(math.sqrt(sum(v * v for v in values)), 6)


def _resolve_reshape_dims(
  current_shape: tuple[int, ...],
  dims: tuple[int | tuple[int, ...] | list[int], ...],
) -> tuple[tuple[int, ...], bool]:
  """Resolve target dimensions for ``reshape``/``view``, inferring ``-1`` dynamically.

  Args:
      current_shape: Existing tensor shape tuple.
      dims: Requested dimension sizes or nested sequences.

  Returns:
      Tuple of ``(resolved_shape, has_dynamic_neg_one)``.
  """
  flat_dims: list[int] = []
  for d in dims:
    if isinstance(d, (tuple, list)):
      flat_dims.extend(int(x) for x in d)
    else:
      flat_dims.append(int(d))
  total = _prod(current_shape)
  has_neg_one = -1 in flat_dims
  if has_neg_one:
    known = 1
    for d in flat_dims:
      if d > 0:
        known *= d
    inferred = max(1, total // max(1, known))
    return tuple(inferred if d == -1 else d for d in flat_dims), True
  return (tuple(flat_dims) if flat_dims else current_shape), False


def _reduce_broadcast_grad(raw: list[float], target_len: int) -> list[float]:
  """Fold broadcasted cotangents of length ``len(raw)`` back into ``target_len``.

  Args:
      raw: Unreduced elementwise gradient values.
      target_len: Length of the operand buffer prior to broadcasting.

  Returns:
      Reduced gradient vector of length ``target_len``.
  """
  if len(raw) == target_len:
    return raw
  acc = [0.0] * target_len
  for idx, val in enumerate(raw):
    acc[idx % target_len] += val
  return acc


def _binary_tensor_op(
  parent_a: OnDeviceTraceTensor,
  other: OnDeviceTraceTensor | int | float,
  op: typing.Callable[[float, float], float],
  grad_self: typing.Callable[[float, float, float], float],
  grad_other: typing.Callable[[float, float, float], float],
) -> OnDeviceTraceTensor:
  """Execute a broadcasted elementwise binary operation with VJP tape links.

  Args:
      parent_a: Left-hand ``OnDeviceTraceTensor``.
      other: Right-hand ``OnDeviceTraceTensor`` or Python scalar.
      op: Forward scalar operator ``(a, b) -> out``.
      grad_self: Left VJP scalar function ``(g, a, b) -> da``.
      grad_other: Right VJP scalar function ``(g, a, b) -> db``.

  Returns:
      Result ``OnDeviceTraceTensor`` linked to reverse-mode tape.
  """
  from ml_switcheroo.core.tensor_core import OnDeviceTraceTensor as TensorCls
  from ml_switcheroo.core.tensor_core import _TapeEntry

  if isinstance(other, TensorCls):
    parent_b = other
    n = max(len(parent_a.data), len(parent_b.data))
    a_vals = [parent_a.data[i % len(parent_a.data)] for i in range(n)]
    b_vals = [parent_b.data[i % len(parent_b.data)] for i in range(n)]
    out_vals = [op(a, b) for a, b in zip(a_vals, b_vals, strict=False)]
    out_shape = parent_a.shape if len(parent_a.data) >= len(parent_b.data) else parent_b.shape

    def vjp_a(g: list[float]) -> list[float]:
      """Compute gradient with respect to left operand.

      Args:
          g: Upstream cotangent vector.

      Returns:
          Gradient vector matching ``parent_a``.
      """
      raw = [grad_self(gv, a, b) for gv, a, b in zip(g, a_vals, b_vals, strict=False)]
      return _reduce_broadcast_grad(raw, len(parent_a.data))

    def vjp_b(g: list[float]) -> list[float]:
      """Compute gradient with respect to right operand.

      Args:
          g: Upstream cotangent vector.

      Returns:
          Gradient vector matching ``parent_b``.
      """
      raw = [grad_other(gv, a, b) for gv, a, b in zip(g, a_vals, b_vals, strict=False)]
      return _reduce_broadcast_grad(raw, len(parent_b.data))

    return TensorCls(
      shape=out_shape,
      data=out_vals,
      dtype=parent_a.dtype,
      requires_grad=parent_a.requires_grad or parent_b.requires_grad,
      tape_entry=_TapeEntry(parents=[(parent_a, vjp_a), (parent_b, vjp_b)]),
      is_dynamic=parent_a.is_dynamic or parent_b.is_dynamic,
      dynamic_reason=parent_a.dynamic_reason or parent_b.dynamic_reason,
    )

  scalar = float(other)
  out_vals = [op(a, scalar) for a in parent_a.data]
  return TensorCls(
    shape=parent_a.shape,
    data=out_vals,
    dtype=parent_a.dtype,
    requires_grad=parent_a.requires_grad,
    tape_entry=_TapeEntry(
      parents=[(parent_a, lambda g: [grad_self(gv, a, scalar) for gv, a in zip(g, parent_a.data, strict=False)])]
    ),
    is_dynamic=parent_a.is_dynamic,
    dynamic_reason=parent_a.dynamic_reason,
  )


__all__ = [
  "_binary_tensor_op",
  "_generate_seeded_values",
  "_l2_norm",
  "_prod",
  "_reduce_broadcast_grad",
  "_resolve_reshape_dims",
]
