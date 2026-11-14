"""Core Differentiable Tensor Type and Reverse-Mode Autodiff Tape for On-Device Tracing.

Defines ``_TapeEntry`` and ``OnDeviceTraceTensor``, delegating elementwise math and
shape calculations to ``ml_switcheroo.core.tensor_ops_math``.
"""

from __future__ import annotations

import math
import typing

from ml_switcheroo.core.tensor_ops_math import (
  _binary_tensor_op,
  _prod,
  _resolve_reshape_dims,
)


class _TapeEntry:
  """Internal reverse-mode autodiff tape entry for ``OnDeviceTraceTensor``."""

  def __init__(
    self,
    parents: list[tuple["OnDeviceTraceTensor", typing.Callable[[list[float]], list[float]]]],
  ) -> None:
    """Initialize tape entry with parent tensor VJP links.

    Args:
        parents: List of ``(parent_tensor, vjp_callable)`` pairs.
    """
    self.parents = parents


class OnDeviceTraceTensor:
  """Pure-Python tensor with reverse-mode autodiff and shape tracking for on-device execution."""

  def __init__(
    self,
    shape: tuple[int, ...],
    data: list[float] | None = None,
    dtype: str = "float32",
    requires_grad: bool = True,
    tape_entry: _TapeEntry | None = None,
    is_dynamic: bool = False,
    dynamic_reason: str | None = None,
  ) -> None:
    """Initialize an ``OnDeviceTraceTensor``.

    Args:
        shape: Concrete tensor dimensions.
        data: Flat list of scalar values matching ``_prod(shape)``.
        dtype: Data type string.
        requires_grad: Whether reverse-mode gradients should be tracked.
        tape_entry: Optional VJP tape link to parent tensors.
        is_dynamic: Whether this tensor's shape depends on runtime values.
        dynamic_reason: Optional explanation for dynamic shape.
    """
    self._shape = tuple(int(d) for d in shape)
    numel = _prod(self._shape)
    if data is None:
      self.data = [0.1] * numel
    elif len(data) == numel:
      self.data = list(data)
    elif len(data) > numel:
      self.data = list(data[:numel])
    else:
      pad = data[-1] if data else 0.0
      self.data = list(data) + [pad] * (numel - len(data))
    self.dtype = dtype
    self.requires_grad = requires_grad
    self._tape = tape_entry
    self.grad: OnDeviceTraceTensor | None = None
    self.is_dynamic = is_dynamic
    self.dynamic_reason = dynamic_reason

  @property
  def shape(self) -> tuple[int, ...]:
    """Return the tensor shape tuple.

    Returns:
        Tuple of integer dimensions.
    """
    return self._shape

  @property
  def ndim(self) -> int:
    """Return the tensor rank.

    Returns:
        Number of dimensions.
    """
    return len(self._shape)

  def item(self) -> float:
    """Return the first scalar value as a Python float.

    Returns:
        Scalar float value.
    """
    return float(self.data[0]) if self.data else 0.0

  def sum(self) -> "OnDeviceTraceTensor":
    """Reduce all elements by summation.

    Returns:
        Scalar ``OnDeviceTraceTensor``.
    """
    n = len(self.data)
    parent = self
    return OnDeviceTraceTensor(
      shape=(),
      data=[sum(self.data)],
      dtype=self.dtype,
      requires_grad=self.requires_grad,
      tape_entry=_TapeEntry(parents=[(parent, lambda g: [g[0]] * n)]),
    )

  def mean(self) -> "OnDeviceTraceTensor":
    """Reduce all elements by arithmetic mean.

    Returns:
        Scalar ``OnDeviceTraceTensor``.
    """
    n = max(1, len(self.data))
    parent = self
    return OnDeviceTraceTensor(
      shape=(),
      data=[sum(self.data) / n],
      dtype=self.dtype,
      requires_grad=self.requires_grad,
      tape_entry=_TapeEntry(parents=[(parent, lambda g: [g[0] / n] * n)]),
    )

  def relu(self) -> "OnDeviceTraceTensor":
    """Apply elementwise ReLU activation.

    Returns:
        Activated ``OnDeviceTraceTensor``.
    """
    parent = self
    out_data = [v if v > 0.0 else 0.0 for v in self.data]
    return OnDeviceTraceTensor(
      shape=self._shape,
      data=out_data,
      dtype=self.dtype,
      requires_grad=self.requires_grad,
      tape_entry=_TapeEntry(
        parents=[(parent, lambda g: [gv if pv > 0.0 else 0.0 for gv, pv in zip(g, parent.data, strict=False)])]
      ),
      is_dynamic=self.is_dynamic,
      dynamic_reason=self.dynamic_reason,
    )

  def sigmoid(self) -> "OnDeviceTraceTensor":
    """Apply elementwise logistic sigmoid activation.

    Returns:
        Activated ``OnDeviceTraceTensor``.
    """
    parent = self
    sig = [1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, v)))) for v in self.data]
    return OnDeviceTraceTensor(
      shape=self._shape,
      data=sig,
      dtype=self.dtype,
      requires_grad=self.requires_grad,
      tape_entry=_TapeEntry(parents=[(parent, lambda g: [gv * s * (1.0 - s) for gv, s in zip(g, sig, strict=False)])]),
    )

  def tanh(self) -> "OnDeviceTraceTensor":
    """Apply elementwise hyperbolic tangent activation.

    Returns:
        Activated ``OnDeviceTraceTensor``.
    """
    parent = self
    th = [math.tanh(v) for v in self.data]
    return OnDeviceTraceTensor(
      shape=self._shape,
      data=th,
      dtype=self.dtype,
      requires_grad=self.requires_grad,
      tape_entry=_TapeEntry(parents=[(parent, lambda g: [gv * (1.0 - t * t) for gv, t in zip(g, th, strict=False)])]),
    )

  def exp(self) -> "OnDeviceTraceTensor":
    """Apply elementwise exponential.

    Returns:
        Transformed ``OnDeviceTraceTensor``.
    """
    parent = self
    ex = [math.exp(max(-20.0, min(20.0, v))) for v in self.data]
    return OnDeviceTraceTensor(
      shape=self._shape,
      data=ex,
      dtype=self.dtype,
      requires_grad=self.requires_grad,
      tape_entry=_TapeEntry(parents=[(parent, lambda g: [gv * ev for gv, ev in zip(g, ex, strict=False)])]),
    )

  def flatten(self, start_dim: int = 1) -> "OnDeviceTraceTensor":
    """Flatten dimensions starting from ``start_dim``.

    Args:
        start_dim: Dimension index from which to flatten.

    Returns:
        Flattened ``OnDeviceTraceTensor``.
    """
    if len(self._shape) <= 1:
      new_shape = self._shape
    else:
      eff_start = max(0, min(len(self._shape) - 1, start_dim))
      new_shape = (*self._shape[:eff_start], _prod(self._shape[eff_start:]))
    parent = self
    return OnDeviceTraceTensor(
      shape=new_shape,
      data=list(self.data),
      dtype=self.dtype,
      requires_grad=self.requires_grad,
      tape_entry=_TapeEntry(parents=[(parent, lambda g: list(g))]),
    )

  def reshape(self, *dims: int | tuple[int, ...] | list[int]) -> "OnDeviceTraceTensor":
    """Reshape tensor to target dimensions, resolving ``-1`` dynamically.

    Args:
        *dims: Target dimension sizes or sequence of dimensions.

    Returns:
        Reshaped ``OnDeviceTraceTensor``.
    """
    resolved, has_neg_one = _resolve_reshape_dims(self._shape, dims)
    parent = self
    return OnDeviceTraceTensor(
      shape=resolved,
      data=list(self.data),
      dtype=self.dtype,
      requires_grad=self.requires_grad,
      tape_entry=_TapeEntry(parents=[(parent, lambda g: list(g))]),
      is_dynamic=has_neg_one or self.is_dynamic,
      dynamic_reason="Dynamic reshape with -1 inferred on-device" if has_neg_one else self.dynamic_reason,
    )

  def view(self, *dims: int | tuple[int, ...] | list[int]) -> "OnDeviceTraceTensor":
    """Return a reshaped view of the tensor.

    Args:
        *dims: Target dimensions.

    Returns:
        Reshaped ``OnDeviceTraceTensor``.
    """
    return self.reshape(*dims)

  def transpose(self, dim0: int = 0, dim1: int = 1) -> "OnDeviceTraceTensor":
    """Swap two dimensions or permute dimensions of the tensor.

    Args:
        dim0: First dimension index.
        dim1: Second dimension index.

    Returns:
        Transposed ``OnDeviceTraceTensor``.
    """
    shape_list = list(self._shape)
    if len(shape_list) >= 2:
      d0 = dim0 % len(shape_list)
      d1 = dim1 % len(shape_list)
      shape_list[d0], shape_list[d1] = shape_list[d1], shape_list[d0]
    parent = self
    return OnDeviceTraceTensor(
      shape=tuple(shape_list),
      data=list(self.data),
      dtype=self.dtype,
      requires_grad=self.requires_grad,
      tape_entry=_TapeEntry(parents=[(parent, lambda g: list(g))]),
    )

  def _binary_op(
    self,
    other: "OnDeviceTraceTensor" | int | float,
    op: typing.Callable[[float, float], float],
    grad_self: typing.Callable[[float, float, float], float],
    grad_other: typing.Callable[[float, float, float], float],
  ) -> "OnDeviceTraceTensor":
    """Execute an elementwise binary operation with VJP recording.

    Args:
        other: Right-hand tensor or scalar.
        op: Forward scalar function.
        grad_self: VJP function for left operand ``(g, a, b) -> da``.
        grad_other: VJP function for right operand ``(g, a, b) -> db``.

    Returns:
        Result ``OnDeviceTraceTensor``.
    """
    return _binary_tensor_op(self, other, op, grad_self, grad_other)

  def __add__(self, other: "OnDeviceTraceTensor" | int | float) -> "OnDeviceTraceTensor":
    """Elementwise addition.

    Args:
        other: Right-hand tensor or scalar.

    Returns:
        Sum tensor.
    """
    return self._binary_op(other, lambda a, b: a + b, lambda g, _a, _b: g, lambda g, _a, _b: g)

  def __radd__(self, other: "OnDeviceTraceTensor" | int | float) -> "OnDeviceTraceTensor":
    """Reverse elementwise addition.

    Args:
        other: Left-hand tensor or scalar.

    Returns:
        Sum tensor.
    """
    return self + other

  def __sub__(self, other: "OnDeviceTraceTensor" | int | float) -> "OnDeviceTraceTensor":
    """Elementwise subtraction.

    Args:
        other: Right-hand tensor or scalar.

    Returns:
        Difference tensor.
    """
    return self._binary_op(other, lambda a, b: a - b, lambda g, _a, _b: g, lambda g, _a, _b: -g)

  def __mul__(self, other: "OnDeviceTraceTensor" | int | float) -> "OnDeviceTraceTensor":
    """Elementwise multiplication.

    Args:
        other: Right-hand tensor or scalar.

    Returns:
        Product tensor.
    """
    return self._binary_op(other, lambda a, b: a * b, lambda g, _a, b: g * b, lambda g, a, _b: g * a)

  def __rmul__(self, other: "OnDeviceTraceTensor" | int | float) -> "OnDeviceTraceTensor":
    """Reverse elementwise multiplication.

    Args:
        other: Left-hand tensor or scalar.

    Returns:
        Product tensor.
    """
    return self * other

  def __truediv__(self, other: "OnDeviceTraceTensor" | int | float) -> "OnDeviceTraceTensor":
    """Elementwise true division.

    Args:
        other: Right-hand tensor or scalar.

    Returns:
        Quotient tensor.
    """
    return self._binary_op(
      other,
      lambda a, b: a / (b if abs(b) > 1e-12 else 1e-12),
      lambda g, _a, b: g / (b if abs(b) > 1e-12 else 1e-12),
      lambda g, a, b: -g * a / ((b * b) if abs(b) > 1e-6 else 1e-12),
    )

  def __gt__(self, other: "OnDeviceTraceTensor" | int | float) -> "OnDeviceTraceTensor":
    """Elementwise greater-than comparison producing a boolean mask tensor.

    Args:
        other: Threshold tensor or scalar.

    Returns:
        Boolean mask ``OnDeviceTraceTensor``.
    """
    thresh = other.item() if isinstance(other, OnDeviceTraceTensor) else float(other)
    return OnDeviceTraceTensor(
      shape=self._shape,
      data=[1.0 if v > thresh else 0.0 for v in self.data],
      dtype="bool",
      requires_grad=False,
    )

  def __lt__(self, other: "OnDeviceTraceTensor" | int | float) -> "OnDeviceTraceTensor":
    """Elementwise less-than comparison producing a boolean mask tensor.

    Args:
        other: Threshold tensor or scalar.

    Returns:
        Boolean mask ``OnDeviceTraceTensor``.
    """
    thresh = other.item() if isinstance(other, OnDeviceTraceTensor) else float(other)
    return OnDeviceTraceTensor(
      shape=self._shape,
      data=[1.0 if v < thresh else 0.0 for v in self.data],
      dtype="bool",
      requires_grad=False,
    )

  def __bool__(self) -> bool:
    """Evaluate truthiness of scalar or tensor condition for control-flow tracing.

    Returns:
        True if any element (for bool dtype) or sum of elements is positive.
    """
    if self.dtype == "bool":
      return any(v > 0.5 for v in self.data)
    return sum(self.data) > 0.0

  def __getitem__(
    self,
    index: (
      "OnDeviceTraceTensor"
      | int
      | slice[int | None, int | None, int | None]
      | tuple[int | slice[int | None, int | None, int | None], ...]
    ),
  ) -> "OnDeviceTraceTensor":
    """Index or boolean-mask a tensor, marking boolean mask outputs as dynamic.

    Args:
        index: Boolean mask tensor, integer index, or slice.

    Returns:
        Indexed ``OnDeviceTraceTensor``.
    """
    parent = self
    if isinstance(index, OnDeviceTraceTensor) and index.dtype == "bool":
      selected_indices = [i for i, m in enumerate(index.data) if m > 0.5 and i < len(self.data)]
      if not selected_indices:
        selected_indices = [0]
      selected_vals = [self.data[i] for i in selected_indices]

      def mask_vjp(g: list[float]) -> list[float]:
        """Scatter gradients back into masked positions.

        Args:
            g: Upstream gradient for masked elements.

        Returns:
            Full gradient vector matching ``parent``.
        """
        out_g = [0.0] * len(parent.data)
        for pos, gv in zip(selected_indices, g, strict=False):
          out_g[pos] += gv
        return out_g

      return OnDeviceTraceTensor(
        shape=(len(selected_vals),),
        data=selected_vals,
        dtype=self.dtype,
        requires_grad=self.requires_grad,
        tape_entry=_TapeEntry(parents=[(parent, mask_vjp)]),
        is_dynamic=True,
        dynamic_reason="Data-dependent boolean mask indexing resolved on-device",
      )

    return OnDeviceTraceTensor(
      shape=self._shape,
      data=list(self.data),
      dtype=self.dtype,
      requires_grad=self.requires_grad,
      tape_entry=_TapeEntry(parents=[(parent, lambda g: list(g))]),
    )

  def backward(self, cotangent: list[float] | None = None) -> None:
    """Run reverse-mode automatic differentiation from this tensor back to all leaf tensors.

    Args:
        cotangent: Optional initial upstream gradient vector; defaults to ones.
    """
    topo: list[OnDeviceTraceTensor] = []
    visited: set[int] = set()

    def build_topo(node: OnDeviceTraceTensor) -> None:
      """Collect nodes in reverse topological order.

      Args:
          node: Current tensor node.
      """
      node_id = id(node)
      if node_id in visited:
        return
      visited.add(node_id)
      if node._tape is not None:
        for parent, _ in node._tape.parents:
          build_topo(parent)
      topo.append(node)

    build_topo(self)

    grads: dict[int, list[float]] = {id(self): list(cotangent) if cotangent is not None else [1.0] * len(self.data)}

    for node in reversed(topo):
      node_grad = grads.get(id(node))
      if node_grad is None:
        continue
      node.grad = OnDeviceTraceTensor(
        shape=node.shape,
        data=list(node_grad),
        dtype=node.dtype,
        requires_grad=False,
      )
      if node._tape is not None:
        for parent, vjp_fn in node._tape.parents:
          if not parent.requires_grad:
            continue
          p_grad = vjp_fn(node_grad)
          pid = id(parent)
          if pid not in grads:
            grads[pid] = p_grad
          else:
            grads[pid] = [a + b for a, b in zip(grads[pid], p_grad, strict=False)]


__all__ = ["OnDeviceTraceTensor", "_TapeEntry"]
