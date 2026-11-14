"""Differentiable Neural Network Kernels and Telemetry Checks for On-Device Tracing.

Provides reverse-mode differentiable ``_matmul_2d``, ``_conv_nd``, ``_compute_loss``,
and ``_check_all_finite`` kernels operating on ``OnDeviceTraceTensor``.
"""

from __future__ import annotations

import math
import types
import typing

from ml_switcheroo.config import LossReduction
from ml_switcheroo.core.tensor_core import OnDeviceTraceTensor, _TapeEntry
from ml_switcheroo.core.tensor_ops_math import _prod


def _matmul_forward(
  x: OnDeviceTraceTensor,
  weight: OnDeviceTraceTensor,
  bias: OnDeviceTraceTensor | None,
  batch_elems: int,
  in_features: int,
  out_features: int,
) -> list[float]:
  """Compute flat forward output values for ``x @ weight.T + bias``.

  Args:
      x: Input activation tensor.
      weight: Weight matrix tensor.
      bias: Optional bias vector tensor.
      batch_elems: Flattened batch element count.
      in_features: Input feature dimension.
      out_features: Output feature dimension.

  Returns:
      Flat output buffer of length ``batch_elems * out_features``.
  """
  out_data: list[float] = []
  for b in range(batch_elems):
    x_offset = b * in_features
    for o in range(out_features):
      w_offset = o * in_features
      acc = 0.0
      for k in range(in_features):
        acc += x.data[(x_offset + k) % len(x.data)] * weight.data[(w_offset + k) % len(weight.data)]
      if bias is not None and bias.data:
        acc += bias.data[o % len(bias.data)]
      out_data.append(acc)
  return out_data


def _make_matmul_vjp_x(
  weight: OnDeviceTraceTensor,
  batch_elems: int,
  in_features: int,
  out_features: int,
) -> typing.Callable[[list[float]], list[float]]:
  """Create the VJP closure with respect to ``x`` for ``_matmul_2d``.

  Args:
      weight: Weight tensor of shape ``(out_features, in_features)``.
      batch_elems: Flattened batch element count.
      in_features: Input feature dimension.
      out_features: Output feature dimension.

  Returns:
      Cotangent mapping ``g -> dx``.
  """

  def vjp_x(g: list[float]) -> list[float]:
    """Compute VJP with respect to input ``x``.

    Args:
        g: Upstream gradient of shape ``(batch_elems, out_features)``.

    Returns:
        Gradient with respect to ``x`` of shape ``(batch_elems, in_features)``.
    """
    dx = [0.0] * (batch_elems * in_features)
    for b in range(batch_elems):
      for k in range(in_features):
        acc = 0.0
        for o in range(out_features):
          acc += g[(b * out_features + o) % len(g)] * weight.data[(o * in_features + k) % len(weight.data)]
        dx[b * in_features + k] = acc
    return dx

  return vjp_x


def _make_matmul_vjp_w(
  x: OnDeviceTraceTensor,
  batch_elems: int,
  in_features: int,
  out_features: int,
) -> typing.Callable[[list[float]], list[float]]:
  """Create the VJP closure with respect to ``weight`` for ``_matmul_2d``.

  Args:
      x: Input activation tensor.
      batch_elems: Flattened batch element count.
      in_features: Input feature dimension.
      out_features: Output feature dimension.

  Returns:
      Cotangent mapping ``g -> dw``.
  """

  def vjp_w(g: list[float]) -> list[float]:
    """Compute VJP with respect to ``weight``.

    Args:
        g: Upstream gradient of shape ``(batch_elems, out_features)``.

    Returns:
        Gradient with respect to ``weight`` of shape ``(out_features, in_features)``.
    """
    dw = [0.0] * (out_features * in_features)
    for o in range(out_features):
      for k in range(in_features):
        acc = 0.0
        for b in range(batch_elems):
          acc += g[(b * out_features + o) % len(g)] * x.data[(b * in_features + k) % len(x.data)]
        dw[o * in_features + k] = acc
    return dw

  return vjp_w


def _matmul_2d(
  x: OnDeviceTraceTensor,
  weight: OnDeviceTraceTensor,
  bias: OnDeviceTraceTensor | None = None,
) -> OnDeviceTraceTensor:
  """Compute differentiable affine transformation ``x @ weight.T + bias``.

  Args:
      x: Input activation tensor of shape ``(..., in_features)``.
      weight: Layer weight tensor of shape ``(out_features, in_features)``.
      bias: Optional bias tensor of shape ``(out_features,)``.

  Returns:
      Output ``OnDeviceTraceTensor`` of shape ``(..., out_features)``.
  """
  in_features = x.shape[-1] if x.shape else 1
  out_features = weight.shape[0] if weight.shape else 1
  batch_elems = max(1, _prod(x.shape[:-1])) if len(x.shape) > 1 else 1
  out_shape = (*x.shape[:-1], out_features) if len(x.shape) > 1 else (out_features,)

  out_data = _matmul_forward(x, weight, bias, batch_elems, in_features, out_features)
  parents: list[tuple[OnDeviceTraceTensor, typing.Callable[[list[float]], list[float]]]] = [
    (x, _make_matmul_vjp_x(weight, batch_elems, in_features, out_features)),
    (weight, _make_matmul_vjp_w(x, batch_elems, in_features, out_features)),
  ]

  if bias is not None:
    bias_ref = bias

    def vjp_b(g: list[float]) -> list[float]:
      """Compute VJP with respect to ``bias``.

      Args:
          g: Upstream gradient.

      Returns:
          Gradient with respect to ``bias``.
      """
      db = [0.0] * out_features
      for b in range(batch_elems):
        for o in range(out_features):
          db[o] += g[(b * out_features + o) % len(g)]
      return db[: len(bias_ref.data)]

    parents.append((bias_ref, vjp_b))

  return OnDeviceTraceTensor(
    shape=out_shape,
    data=out_data,
    dtype=x.dtype,
    requires_grad=True,
    tape_entry=_TapeEntry(parents=parents),
    is_dynamic=x.is_dynamic,
    dynamic_reason=x.dynamic_reason,
  )


def _conv_forward(
  x: OnDeviceTraceTensor,
  weight: OnDeviceTraceTensor,
  bias: OnDeviceTraceTensor | None,
  batch: int,
  in_channels: int,
  out_channels: int,
  spatial_elems: int,
) -> list[float]:
  """Compute flat forward output values for N-D convolution tracing.

  Args:
      x: Input activation tensor.
      weight: Convolution kernel tensor.
      bias: Optional convolution bias tensor.
      batch: Batch dimension size.
      in_channels: Input channel count.
      out_channels: Output channel count.
      spatial_elems: Flattened spatial element count.

  Returns:
      Flat output buffer of length ``batch * out_channels * spatial_elems``.
  """
  out_data: list[float] = []
  for b in range(batch):
    for oc in range(out_channels):
      b_val = bias.data[oc % len(bias.data)] if (bias is not None and bias.data) else 0.0
      for s in range(spatial_elems):
        acc = b_val
        for ic in range(in_channels):
          x_idx = ((b * in_channels + ic) * spatial_elems + s) % len(x.data)
          w_idx = (oc * in_channels + ic) % len(weight.data)
          acc += x.data[x_idx] * weight.data[w_idx]
        out_data.append(acc)
  return out_data


def _make_conv_vjp_x(
  x: OnDeviceTraceTensor,
  weight: OnDeviceTraceTensor,
  batch: int,
  in_channels: int,
  out_channels: int,
  spatial_elems: int,
) -> typing.Callable[[list[float]], list[float]]:
  """Create the VJP closure with respect to ``x`` for ``_conv_nd``.

  Args:
      x: Input activation tensor.
      weight: Convolution kernel tensor.
      batch: Batch dimension size.
      in_channels: Input channel count.
      out_channels: Output channel count.
      spatial_elems: Flattened spatial element count.

  Returns:
      Cotangent mapping ``g -> dx``.
  """

  def vjp_x(g: list[float]) -> list[float]:
    """Compute VJP with respect to convolution input ``x``.

    Args:
        g: Upstream gradient vector.

    Returns:
        Input gradient vector matching ``x``.
    """
    dx = [0.0] * len(x.data)
    for b in range(batch):
      for ic in range(in_channels):
        for s in range(spatial_elems):
          acc = 0.0
          for oc in range(out_channels):
            g_idx = ((b * out_channels + oc) * spatial_elems + s) % len(g)
            w_idx = (oc * in_channels + ic) % len(weight.data)
            acc += g[g_idx] * weight.data[w_idx]
          dx[((b * in_channels + ic) * spatial_elems + s) % len(dx)] = acc
    return dx

  return vjp_x


def _make_conv_vjp_w(
  x: OnDeviceTraceTensor,
  weight: OnDeviceTraceTensor,
  batch: int,
  in_channels: int,
  out_channels: int,
  spatial_elems: int,
) -> typing.Callable[[list[float]], list[float]]:
  """Create the VJP closure with respect to ``weight`` for ``_conv_nd``.

  Args:
      x: Input activation tensor.
      weight: Convolution kernel tensor.
      batch: Batch dimension size.
      in_channels: Input channel count.
      out_channels: Output channel count.
      spatial_elems: Flattened spatial element count.

  Returns:
      Cotangent mapping ``g -> dw``.
  """

  def vjp_w(g: list[float]) -> list[float]:
    """Compute VJP with respect to convolution ``weight``.

    Args:
        g: Upstream gradient vector.

    Returns:
        Weight gradient vector matching ``weight``.
    """
    dw = [0.0] * len(weight.data)
    for oc in range(out_channels):
      for ic in range(in_channels):
        acc = 0.0
        for b in range(batch):
          for s in range(spatial_elems):
            g_idx = ((b * out_channels + oc) * spatial_elems + s) % len(g)
            x_idx = ((b * in_channels + ic) * spatial_elems + s) % len(x.data)
            acc += g[g_idx] * x.data[x_idx]
        dw[(oc * in_channels + ic) % len(dw)] += acc
    return dw

  return vjp_w


def _conv_nd(
  x: OnDeviceTraceTensor,
  weight: OnDeviceTraceTensor,
  bias: OnDeviceTraceTensor | None,
  out_channels: int,
) -> OnDeviceTraceTensor:
  """Compute differentiable N-D convolution projection for shape and gradient tracing.

  Args:
      x: Input activation tensor ``(B, C_in, ...)``.
      weight: Convolution kernel tensor.
      bias: Optional convolution bias tensor.
      out_channels: Number of output channels.

  Returns:
      Convolved ``OnDeviceTraceTensor`` of shape ``(B, out_channels, ...)``.
  """
  batch = x.shape[0] if x.shape else 1
  spatial = x.shape[2:] if len(x.shape) > 2 else (8,)
  out_shape = (batch, out_channels, *spatial)
  spatial_elems = _prod(spatial)
  in_channels = x.shape[1] if len(x.shape) > 1 else 1

  out_data = _conv_forward(x, weight, bias, batch, in_channels, out_channels, spatial_elems)
  parents: list[tuple[OnDeviceTraceTensor, typing.Callable[[list[float]], list[float]]]] = [
    (x, _make_conv_vjp_x(x, weight, batch, in_channels, out_channels, spatial_elems)),
    (weight, _make_conv_vjp_w(x, weight, batch, in_channels, out_channels, spatial_elems)),
  ]
  if bias is not None:
    bias_ref = bias

    def vjp_b(g: list[float]) -> list[float]:
      """Compute VJP with respect to convolution ``bias``.

      Args:
          g: Upstream gradient vector.

      Returns:
          Bias gradient vector.
      """
      db = [0.0] * len(bias_ref.data)
      for b in range(batch):
        for oc in range(out_channels):
          for s in range(spatial_elems):
            db[oc % len(db)] += g[((b * out_channels + oc) * spatial_elems + s) % len(g)]
      return db

    parents.append((bias_ref, vjp_b))

  return OnDeviceTraceTensor(
    shape=out_shape,
    data=out_data,
    dtype=x.dtype,
    requires_grad=True,
    tape_entry=_TapeEntry(parents=parents),
  )


def _compute_loss(output: OnDeviceTraceTensor, reduction: LossReduction) -> OnDeviceTraceTensor:
  """Compute scalar loss tensor from the forward output according to ``reduction``.

  Args:
      output: Forward pass output ``OnDeviceTraceTensor``.
      reduction: Configured ``LossReduction`` (``SUM``, ``MEAN``, or ``L2``).

  Returns:
      Scalar loss ``OnDeviceTraceTensor``.
  """
  if reduction == LossReduction.MEAN:
    return output.mean()
  if reduction == LossReduction.L2:
    return (output * output).sum() * 0.5
  return output.sum()


@typing.runtime_checkable
class _ChexFiniteProtocol(typing.Protocol):
  """Protocol for ``zero_chex`` module providing ``assert_tree_all_finite``."""

  def assert_tree_all_finite(self, tree: list[float]) -> None:
    """Assert all values in ``tree`` are finite.

    Args:
        tree: Flat list of floats to verify.
    """
    ...


def _check_all_finite(
  activations: typing.Sequence[OnDeviceTraceTensor],
  params: typing.Mapping[str, OnDeviceTraceTensor],
  sandbox_mods: typing.Mapping[str, types.ModuleType],
) -> bool:
  """Verify that all forward activations and backward gradients are finite (``zero_chex``).

  Args:
      activations: Traced activation tensors.
      params: Model parameter tensors.
      sandbox_mods: Active ``zero-*`` modules in ``sys.modules``.

  Returns:
      True if all values and gradients are finite (no NaN or Inf).
  """
  all_vals: list[float] = []
  for act in activations:
    all_vals.extend(act.data)
    if act.grad is not None:
      all_vals.extend(act.grad.data)
  for p in params.values():
    all_vals.extend(p.data)
    if p.grad is not None:
      all_vals.extend(p.grad.data)

  chex_mod = sandbox_mods.get("chex")
  if isinstance(chex_mod, _ChexFiniteProtocol):
    try:
      chex_mod.assert_tree_all_finite(all_vals)
    except (AssertionError, ValueError, TypeError, RuntimeError):
      return False

  return all(math.isfinite(v) for v in all_vals)


__all__ = [
  "_ChexFiniteProtocol",
  "_check_all_finite",
  "_compute_loss",
  "_conv_nd",
  "_matmul_2d",
]
