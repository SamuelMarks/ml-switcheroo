"""Pure-Python Differentiable Tensor and Operators Facade for On-Device Tracing.

Re-exports ``OnDeviceTraceTensor`` and reverse-mode autodiff kernels implemented across:
- ``ml_switcheroo.core.tensor_core``
- ``ml_switcheroo.core.tensor_ops_math``
- ``ml_switcheroo.core.tensor_ops_nn``
"""

from __future__ import annotations

from ml_switcheroo.core.tensor_core import OnDeviceTraceTensor, _TapeEntry
from ml_switcheroo.core.tensor_ops_math import (
  _generate_seeded_values,
  _l2_norm,
  _prod,
)
from ml_switcheroo.core.tensor_ops_nn import (
  _check_all_finite,
  _ChexFiniteProtocol,
  _compute_loss,
  _conv_nd,
  _matmul_2d,
)

__all__ = [
  "OnDeviceTraceTensor",
  "_ChexFiniteProtocol",
  "_TapeEntry",
  "_check_all_finite",
  "_compute_loss",
  "_conv_nd",
  "_generate_seeded_values",
  "_l2_norm",
  "_matmul_2d",
  "_prod",
]
