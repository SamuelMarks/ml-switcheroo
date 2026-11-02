"""Pure-Python Differentiable Tensor and Operators for On-Device Tracing.

This module implements `OnDeviceTraceTensor` and reverse-mode automatic
differentiation primitives (`_matmul_2d`, `_conv_nd`) used by `runtime_tracer`
to execute forward and backward passes on-device when tracing model ASTs.
"""

from __future__ import annotations

import math
import random
import types
import typing

from ml_switcheroo.config import LossReduction


def _prod(shape: typing.Sequence[int]) -> int:
  """Compute the product of dimensions in a shape sequence.

  Args:
      shape: Sequence of integer dimension sizes.

  Returns:
      Total number of elements (minimum 1 for scalar `()`).
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
      dtype: Target dtype string ('int32', 'float32', 'float16').
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


class _TapeEntry:
  """Internal reverse-mode autodiff tape entry for `OnDeviceTraceTensor`."""

  def __init__(
    self,
    parents: list[tuple["OnDeviceTraceTensor", typing.Callable[[list[float]], list[float]]]],
  ) -> None:
    """Initialize tape entry with parent tensor VJP links.

    Args:
        parents: List of `(parent_tensor, vjp_callable)` pairs.
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
    """Initialize an `OnDeviceTraceTensor`.

    Args:
        shape: Concrete tensor dimensions.
        data: Flat list of scalar values matching `_prod(shape)`.
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
        Scalar `OnDeviceTraceTensor`.
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
        Scalar `OnDeviceTraceTensor`.
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
        Activated `OnDeviceTraceTensor`.
    """
    parent = self
    out_data = [v if v > 0.0 else 0.0 for v in self.data]
    return OnDeviceTraceTensor(
      shape=self._shape,
      data=out_data,
      dtype=self.dtype,
      requires_grad=self.requires_grad,
      tape_entry=_TapeEntry(parents=[(parent, lambda g: [gv if pv > 0.0 else 0.0 for gv, pv in zip(g, parent.data)])]),
      is_dynamic=self.is_dynamic,
      dynamic_reason=self.dynamic_reason,
    )

  def sigmoid(self) -> "OnDeviceTraceTensor":
    """Apply elementwise logistic sigmoid activation.

    Returns:
        Activated `OnDeviceTraceTensor`.
    """
    parent = self
    sig = [1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, v)))) for v in self.data]
    return OnDeviceTraceTensor(
      shape=self._shape,
      data=sig,
      dtype=self.dtype,
      requires_grad=self.requires_grad,
      tape_entry=_TapeEntry(parents=[(parent, lambda g: [gv * s * (1.0 - s) for gv, s in zip(g, sig)])]),
    )

  def tanh(self) -> "OnDeviceTraceTensor":
    """Apply elementwise hyperbolic tangent activation.

    Returns:
        Activated `OnDeviceTraceTensor`.
    """
    parent = self
    th = [math.tanh(v) for v in self.data]
    return OnDeviceTraceTensor(
      shape=self._shape,
      data=th,
      dtype=self.dtype,
      requires_grad=self.requires_grad,
      tape_entry=_TapeEntry(parents=[(parent, lambda g: [gv * (1.0 - t * t) for gv, t in zip(g, th)])]),
    )

  def exp(self) -> "OnDeviceTraceTensor":
    """Apply elementwise exponential.

    Returns:
        Transformed `OnDeviceTraceTensor`.
    """
    parent = self
    ex = [math.exp(max(-20.0, min(20.0, v))) for v in self.data]
    return OnDeviceTraceTensor(
      shape=self._shape,
      data=ex,
      dtype=self.dtype,
      requires_grad=self.requires_grad,
      tape_entry=_TapeEntry(parents=[(parent, lambda g: [gv * ev for gv, ev in zip(g, ex)])]),
    )

  def flatten(self, start_dim: int = 1) -> "OnDeviceTraceTensor":
    """Flatten dimensions starting from `start_dim`.

    Args:
        start_dim: Dimension index from which to flatten.

    Returns:
        Flattened `OnDeviceTraceTensor`.
    """
    if len(self._shape) <= 1:
      new_shape = self._shape
    else:
      eff_start = max(0, min(len(self._shape) - 1, start_dim))
      prefix = self._shape[:eff_start]
      rest = _prod(self._shape[eff_start:])
      new_shape = (*prefix, rest)
    parent = self
    return OnDeviceTraceTensor(
      shape=new_shape,
      data=list(self.data),
      dtype=self.dtype,
      requires_grad=self.requires_grad,
      tape_entry=_TapeEntry(parents=[(parent, lambda g: list(g))]),
    )

  def reshape(self, *dims: int | tuple[int, ...] | list[int]) -> "OnDeviceTraceTensor":
    """Reshape tensor to target dimensions, resolving `-1` dynamically.

    Args:
        *dims: Target dimension sizes or sequence of dimensions.

    Returns:
        Reshaped `OnDeviceTraceTensor`.
    """
    flat_dims: list[int] = []
    for d in dims:
      if isinstance(d, (tuple, list)):
        flat_dims.extend(int(x) for x in d)
      else:
        flat_dims.append(int(d))
    total = _prod(self._shape)
    has_neg_one = -1 in flat_dims
    if has_neg_one:
      known = 1
      for d in flat_dims:
        if d > 0:
          known *= d
      inferred = max(1, total // max(1, known))
      resolved = tuple(inferred if d == -1 else d for d in flat_dims)
    else:
      resolved = tuple(flat_dims) if flat_dims else self._shape
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
        Reshaped `OnDeviceTraceTensor`.
    """
    return self.reshape(*dims)

  def transpose(self, dim0: int = 0, dim1: int = 1) -> "OnDeviceTraceTensor":
    """Swap two dimensions or permute dimensions of the tensor.

    Args:
        dim0: First dimension index.
        dim1: Second dimension index.

    Returns:
        Transposed `OnDeviceTraceTensor`.
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
        grad_self: VJP function for left operand `(g, a, b) -> da`.
        grad_other: VJP function for right operand `(g, a, b) -> db`.

    Returns:
        Result `OnDeviceTraceTensor`.
    """
    parent_a = self
    if isinstance(other, OnDeviceTraceTensor):
      parent_b = other
      n = max(len(parent_a.data), len(parent_b.data))
      a_vals = [parent_a.data[i % len(parent_a.data)] for i in range(n)]
      b_vals = [parent_b.data[i % len(parent_b.data)] for i in range(n)]
      out_vals = [op(a, b) for a, b in zip(a_vals, b_vals)]
      out_shape = parent_a.shape if len(parent_a.data) >= len(parent_b.data) else parent_b.shape

      def vjp_a(g: list[float]) -> list[float]:
        """Compute gradient with respect to left operand.

        Args:
            g: Upstream cotangent vector.

        Returns:
            Gradient vector matching `parent_a`.
        """
        raw = [grad_self(gv, a, b) for gv, a, b in zip(g, a_vals, b_vals)]
        target_len = len(parent_a.data)
        if len(raw) == target_len:
          return raw
        acc = [0.0] * target_len
        for idx, val in enumerate(raw):
          acc[idx % target_len] += val
        return acc

      def vjp_b(g: list[float]) -> list[float]:
        """Compute gradient with respect to right operand.

        Args:
            g: Upstream cotangent vector.

        Returns:
            Gradient vector matching `parent_b`.
        """
        raw = [grad_other(gv, a, b) for gv, a, b in zip(g, a_vals, b_vals)]
        target_len = len(parent_b.data)
        if len(raw) == target_len:
          return raw
        acc = [0.0] * target_len
        for idx, val in enumerate(raw):
          acc[idx % target_len] += val
        return acc

      return OnDeviceTraceTensor(
        shape=out_shape,
        data=out_vals,
        dtype=self.dtype,
        requires_grad=parent_a.requires_grad or parent_b.requires_grad,
        tape_entry=_TapeEntry(parents=[(parent_a, vjp_a), (parent_b, vjp_b)]),
        is_dynamic=parent_a.is_dynamic or parent_b.is_dynamic,
        dynamic_reason=parent_a.dynamic_reason or parent_b.dynamic_reason,
      )

    scalar = float(other)
    out_vals = [op(a, scalar) for a in parent_a.data]
    return OnDeviceTraceTensor(
      shape=parent_a.shape,
      data=out_vals,
      dtype=self.dtype,
      requires_grad=parent_a.requires_grad,
      tape_entry=_TapeEntry(
        parents=[(parent_a, lambda g: [grad_self(gv, a, scalar) for gv, a in zip(g, parent_a.data)])]
      ),
      is_dynamic=parent_a.is_dynamic,
      dynamic_reason=parent_a.dynamic_reason,
    )

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
    return self.__add__(other)

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
    return self.__mul__(other)

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
        Boolean mask `OnDeviceTraceTensor`.
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
        Boolean mask `OnDeviceTraceTensor`.
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
        Indexed `OnDeviceTraceTensor`.
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
            Full gradient vector matching `parent`.
        """
        out_g = [0.0] * len(parent.data)
        for pos, gv in zip(selected_indices, g):
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
            grads[pid] = [a + b for a, b in zip(grads[pid], p_grad)]


def _matmul_2d(
  x: OnDeviceTraceTensor,
  weight: OnDeviceTraceTensor,
  bias: OnDeviceTraceTensor | None = None,
) -> OnDeviceTraceTensor:
  """Compute differentiable affine transformation `x @ weight.T + bias`.

  Args:
      x: Input activation tensor of shape `(..., in_features)`.
      weight: Layer weight tensor of shape `(out_features, in_features)`.
      bias: Optional bias tensor of shape `(out_features,)`.

  Returns:
      Output `OnDeviceTraceTensor` of shape `(..., out_features)`.
  """
  in_features = x.shape[-1] if x.shape else 1
  out_features = weight.shape[0] if weight.shape else 1
  batch_elems = max(1, _prod(x.shape[:-1])) if len(x.shape) > 1 else 1
  out_shape = (*x.shape[:-1], out_features) if len(x.shape) > 1 else (out_features,)

  out_data: list[float] = []
  for b in range(batch_elems):
    x_offset = b * in_features
    for o in range(out_features):
      w_offset = o * in_features
      acc = 0.0
      for k in range(in_features):
        xv = x.data[(x_offset + k) % len(x.data)]
        wv = weight.data[(w_offset + k) % len(weight.data)]
        acc += xv * wv
      if bias is not None and bias.data:
        acc += bias.data[o % len(bias.data)]
      out_data.append(acc)

  def vjp_x(g: list[float]) -> list[float]:
    """Compute VJP with respect to input `x`.

    Args:
        g: Upstream gradient of shape `(batch_elems, out_features)`.

    Returns:
        Gradient with respect to `x` of shape `(batch_elems, in_features)`.
    """
    dx = [0.0] * (batch_elems * in_features)
    for b in range(batch_elems):
      for k in range(in_features):
        acc = 0.0
        for o in range(out_features):
          gv = g[(b * out_features + o) % len(g)]
          wv = weight.data[(o * in_features + k) % len(weight.data)]
          acc += gv * wv
        dx[b * in_features + k] = acc
    return dx

  def vjp_w(g: list[float]) -> list[float]:
    """Compute VJP with respect to `weight`.

    Args:
        g: Upstream gradient of shape `(batch_elems, out_features)`.

    Returns:
        Gradient with respect to `weight` of shape `(out_features, in_features)`.
    """
    dw = [0.0] * (out_features * in_features)
    for o in range(out_features):
      for k in range(in_features):
        acc = 0.0
        for b in range(batch_elems):
          gv = g[(b * out_features + o) % len(g)]
          xv = x.data[(b * in_features + k) % len(x.data)]
          acc += gv * xv
        dw[o * in_features + k] = acc
    return dw

  parents: list[tuple[OnDeviceTraceTensor, typing.Callable[[list[float]], list[float]]]] = [
    (x, vjp_x),
    (weight, vjp_w),
  ]

  if bias is not None:
    bias_ref = bias

    def vjp_b(g: list[float]) -> list[float]:
      """Compute VJP with respect to `bias`.

      Args:
          g: Upstream gradient.

      Returns:
          Gradient with respect to `bias`.
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


def _conv_nd(
  x: OnDeviceTraceTensor,
  weight: OnDeviceTraceTensor,
  bias: OnDeviceTraceTensor | None,
  out_channels: int,
) -> OnDeviceTraceTensor:
  """Compute differentiable N-D convolution projection for shape and gradient tracing.

  Args:
      x: Input activation tensor `(B, C_in, ...)`.
      weight: Convolution kernel tensor.
      bias: Optional convolution bias tensor.
      out_channels: Number of output channels.

  Returns:
      Convolved `OnDeviceTraceTensor` of shape `(B, out_channels, ...)`.
  """
  batch = x.shape[0] if x.shape else 1
  spatial = x.shape[2:] if len(x.shape) > 2 else (8,)
  out_shape = (batch, out_channels, *spatial)
  spatial_elems = _prod(spatial)
  in_channels = x.shape[1] if len(x.shape) > 1 else 1

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

  def vjp_x(g: list[float]) -> list[float]:
    """Compute VJP with respect to convolution input `x`.

    Args:
        g: Upstream gradient vector.

    Returns:
        Input gradient vector matching `x`.
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

  def vjp_w(g: list[float]) -> list[float]:
    """Compute VJP with respect to convolution `weight`.

    Args:
        g: Upstream gradient vector.

    Returns:
        Weight gradient vector matching `weight`.
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

  parents: list[tuple[OnDeviceTraceTensor, typing.Callable[[list[float]], list[float]]]] = [
    (x, vjp_x),
    (weight, vjp_w),
  ]
  if bias is not None:
    bias_ref = bias

    def vjp_b(g: list[float]) -> list[float]:
      """Compute VJP with respect to convolution `bias`.

      Args:
          g: Upstream gradient vector.

      Returns:
          Bias gradient vector.
      """
      db = [0.0] * len(bias_ref.data)
      for b in range(batch):
        for oc in range(out_channels):
          for s in range(spatial_elems):
            g_idx = ((b * out_channels + oc) * spatial_elems + s) % len(g)
            db[oc % len(db)] += g[g_idx]
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
  """Compute scalar loss tensor from the forward output according to `reduction`.

  Args:
      output: Forward pass output `OnDeviceTraceTensor`.
      reduction: Configured `LossReduction` (`SUM`, `MEAN`, or `L2`).

  Returns:
      Scalar loss `OnDeviceTraceTensor`.
  """
  if reduction == LossReduction.MEAN:
    return output.mean()
  if reduction == LossReduction.L2:
    return (output * output).sum() * 0.5
  return output.sum()


def _l2_norm(values: typing.Sequence[float]) -> float:
  """Compute the Euclidean (L2) norm of a sequence of floats.

  Args:
      values: Sequence of scalar float values.

  Returns:
      L2 norm rounded to 6 decimal places.
  """
  return round(math.sqrt(sum(v * v for v in values)), 6)


def _check_all_finite(
  activations: typing.Sequence[OnDeviceTraceTensor],
  params: typing.Mapping[str, OnDeviceTraceTensor],
  sandbox_mods: typing.Mapping[str, types.ModuleType],
) -> bool:
  """Verify that all forward activations and backward gradients are finite (`zero_chex`).

  Args:
      activations: Traced activation tensors.
      params: Model parameter tensors.
      sandbox_mods: Active `zero-*` modules in `sys.modules`.

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
  if chex_mod is not None and hasattr(chex_mod, "assert_tree_all_finite"):
    try:
      getattr(chex_mod, "assert_tree_all_finite")(all_vals)
    except Exception:
      return False

  return all(math.isfinite(v) for v in all_vals)
