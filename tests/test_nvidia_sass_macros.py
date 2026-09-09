"""Docstring."""

from typing import Callable

from ml_switcheroo.core.compiler.backends.nvidia_sass.macros import (
  RegisterAllocatorProtocol,
  expand_avgpool2d,
  expand_batchnorm2d,
  expand_conv2d,
  expand_conv3d,
  expand_crossentropyloss,
  expand_dropout,
  expand_flatten,
  expand_gelu,
  expand_linear,
  expand_maxpool2d,
  expand_mean,
  expand_mseloss,
  expand_relu,
  expand_reshape,
  expand_sigmoid,
  expand_tanh,
)
from ml_switcheroo.core.compiler.backends.nvidia_sass.macros_extra import (
  _make_generic_expand,
  expand_adam,
  expand_adaptivepool,
  expand_conv1d,
  expand_conv_general_dilated,
  expand_convtranspose,
  expand_depthwiseconv2d,
  expand_gru,
  expand_l,
  expand_lstm,
  expand_multiheadattention,
  expand_pool1d,
  expand_pool3d,
  expand_rnn,
  expand_transformer,
  expand_transformerdecoder,
  expand_transformerencoder,
  expand_transpose,
  expand_variable,
)
from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassRegister


class DummyAllocator(RegisterAllocatorProtocol):
  """Docstring."""

  def get_register(self, var_name: str) -> NvidiaSassRegister:
    """Docstring."""
    return NvidiaSassRegister(name="R0")

  def allocate_temp(self) -> NvidiaSassRegister:
    """Docstring."""
    return NvidiaSassRegister(name="R1")

  def allocate_predicate(self) -> NvidiaSassRegister:
    """Docstring."""
    return NvidiaSassRegister(name="P0")


def test_nvidia_sass_macros() -> None:
  """Docstring."""
  alloc: DummyAllocator = DummyAllocator()

  # macros.py
  assert len(expand_conv2d(alloc, "conv", {"k": 3})) > 0
  assert len(expand_linear(alloc, "lin", {"d_in": 10})) > 0
  assert len(expand_linear(alloc, "lin", {"bias": True})) > 0
  assert len(expand_flatten(alloc, "flat", {})) > 0
  assert len(expand_reshape(alloc, "resh", {})) > 0
  assert len(expand_conv3d(alloc, "conv3d", {})) > 0
  assert len(expand_dropout(alloc, "drop", {})) > 0
  assert len(expand_relu(alloc, "relu", {})) > 0
  assert len(expand_mean(alloc, "mean", {})) > 0
  assert len(expand_avgpool2d(alloc, "avgpool", {})) > 0
  assert len(expand_maxpool2d(alloc, "maxpool", {})) > 0
  assert len(expand_batchnorm2d(alloc, "bn", {})) > 0
  assert len(expand_sigmoid(alloc, "sig", {})) > 0
  assert len(expand_tanh(alloc, "tanh", {})) > 0
  assert len(expand_gelu(alloc, "gelu", {})) > 0
  assert len(expand_mseloss(alloc, "mseloss", {})) > 0
  assert len(expand_mseloss(alloc, "mseloss", {"reduction": "sum"})) > 0
  assert len(expand_mseloss(alloc, "mseloss", {"elements": 0, "reduction": "mean"})) > 0
  assert len(expand_crossentropyloss(alloc, "ce", {})) > 0

  # macros_extra.py
  assert len(expand_rnn(alloc, "rnn", {})) > 0
  assert len(expand_lstm(alloc, "lstm", {})) > 0
  assert len(expand_gru(alloc, "gru", {})) > 0
  assert len(expand_multiheadattention(alloc, "mha", {})) > 0
  assert len(expand_transformer(alloc, "tx", {})) > 0
  assert len(expand_transformerencoder(alloc, "txe", {})) > 0
  assert len(expand_transformerdecoder(alloc, "txd", {})) > 0
  assert len(expand_conv1d(alloc, "c1d", {})) > 0
  assert len(expand_depthwiseconv2d(alloc, "dc2d", {})) > 0
  assert len(expand_convtranspose(alloc, "ct", {})) > 0
  assert len(expand_pool1d(alloc, "p1d", {})) > 0
  assert len(expand_pool3d(alloc, "p3d", {})) > 0
  assert len(expand_adaptivepool(alloc, "ap", {})) > 0
  assert len(expand_variable(alloc, "var", {})) > 0
  assert len(expand_transpose(alloc, "trans", {})) > 0
  assert len(expand_conv_general_dilated(alloc, "cgen", {})) > 0
  assert len(expand_adam(alloc, "adam", {})) > 0
  assert len(expand_l(alloc, "l", {})) > 0

  gen: Callable = _make_generic_expand("generic")
  assert len(gen(alloc, "g", {})) > 0
