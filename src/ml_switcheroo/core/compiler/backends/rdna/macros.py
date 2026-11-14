"""Facade for RDNA macros."""

from ml_switcheroo.core.compiler.backends.rdna.activation import (
  expand_gelu,
  expand_relu,
  expand_sigmoid,
  expand_silu,
  expand_softmax,
  expand_tanh,
)
from ml_switcheroo.core.compiler.backends.rdna.conv import (
  expand_conv2d,
  expand_conv3d,
  expand_conv_general_dilated,
)
from ml_switcheroo.core.compiler.backends.rdna.gemm import (
  expand_generic_linalg,
  expand_linear,
)
from ml_switcheroo.core.compiler.backends.rdna.misc import (
  expand_adam,
  expand_dropout,
  expand_flatten,
  expand_l,
  expand_reshape,
  expand_transpose,
  expand_variable,
)
from ml_switcheroo.core.compiler.backends.rdna.reduction import (
  expand_batchnorm,
  expand_generic_norm,
  expand_generic_reduction,
  expand_layernorm,
  expand_rmsnorm,
)
from ml_switcheroo.core.compiler.isa.allocator import RegisterAllocatorProtocol

__all__ = [
  "RegisterAllocatorProtocol",
  "expand_adam",
  "expand_batchnorm",
  "expand_conv2d",
  "expand_conv3d",
  "expand_conv_general_dilated",
  "expand_dropout",
  "expand_flatten",
  "expand_gelu",
  "expand_generic_linalg",
  "expand_generic_norm",
  "expand_generic_reduction",
  "expand_l",
  "expand_layernorm",
  "expand_linear",
  "expand_relu",
  "expand_reshape",
  "expand_rmsnorm",
  "expand_sigmoid",
  "expand_silu",
  "expand_softmax",
  "expand_tanh",
  "expand_transpose",
  "expand_variable",
]
