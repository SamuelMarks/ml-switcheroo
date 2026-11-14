"""Facade for NVIDIA SASS macros."""

from ml_switcheroo.core.compiler.backends.nvidia_sass.activation import (
  expand_gelu,
  expand_relu,
  expand_sigmoid,
  expand_tanh,
)
from ml_switcheroo.core.compiler.backends.nvidia_sass.conv import (
  expand_adaptivepool,
  expand_avgpool2d,
  expand_conv1d,
  expand_conv2d,
  expand_conv3d,
  expand_conv_general_dilated,
  expand_convtranspose,
  expand_depthwiseconv2d,
  expand_maxpool2d,
  expand_pool1d,
  expand_pool3d,
)
from ml_switcheroo.core.compiler.backends.nvidia_sass.gemm import (
  expand_linear,
  expand_multiheadattention,
  expand_transformer,
  expand_transformerdecoder,
  expand_transformerencoder,
)
from ml_switcheroo.core.compiler.backends.nvidia_sass.misc import (
  expand_adam,
  expand_crossentropyloss,
  expand_dropout,
  expand_flatten,
  expand_gru,
  expand_l,
  expand_lstm,
  expand_mseloss,
  expand_reshape,
  expand_rnn,
  expand_transpose,
  expand_variable,
)
from ml_switcheroo.core.compiler.backends.nvidia_sass.normalization import (
  expand_batchnorm2d,
  expand_mean,
)
from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassRegister
from ml_switcheroo.core.compiler.isa.allocator import RegisterAllocatorProtocol

__all__ = [
  "NvidiaSassRegister",
  "RegisterAllocatorProtocol",
  "expand_adam",
  "expand_adaptivepool",
  "expand_avgpool2d",
  "expand_batchnorm2d",
  "expand_conv1d",
  "expand_conv2d",
  "expand_conv3d",
  "expand_conv_general_dilated",
  "expand_convtranspose",
  "expand_crossentropyloss",
  "expand_depthwiseconv2d",
  "expand_dropout",
  "expand_flatten",
  "expand_gelu",
  "expand_gru",
  "expand_l",
  "expand_linear",
  "expand_lstm",
  "expand_maxpool2d",
  "expand_mean",
  "expand_mseloss",
  "expand_multiheadattention",
  "expand_pool1d",
  "expand_pool3d",
  "expand_relu",
  "expand_reshape",
  "expand_rnn",
  "expand_sigmoid",
  "expand_tanh",
  "expand_transformer",
  "expand_transformerdecoder",
  "expand_transformerencoder",
  "expand_transpose",
  "expand_variable",
]
