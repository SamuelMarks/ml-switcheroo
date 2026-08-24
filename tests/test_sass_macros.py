"""Docstring."""

from ml_switcheroo.core.compiler.backends.sass.macros import (
  expand_conv2d,
  expand_linear,
  expand_flatten,
  expand_reshape,
  expand_conv3d,
  expand_dropout,
  expand_relu,
  expand_mean,
  expand_avgpool2d,
  expand_maxpool2d,
  expand_batchnorm2d,
  expand_sigmoid,
  expand_tanh,
  expand_gelu,
  expand_mseloss,
  expand_crossentropyloss,
)
from ml_switcheroo.core.compiler.backends.sass.macros_extra import (
  expand_rnn,
  expand_lstm,
  expand_gru,
  expand_multiheadattention,
  expand_transformer,
  expand_transformerencoder,
  expand_transformerdecoder,
  expand_conv1d,
  expand_depthwiseconv2d,
  expand_convtranspose,
  expand_pool1d,
  expand_pool3d,
  expand_adaptivepool,
  expand_variable,
  expand_transpose,
  expand_conv_general_dilated,
  expand_adam,
  expand_l,
  _make_generic_expand,
)

from ml_switcheroo.core.compiler.frontends.sass.cst import SassRegister


class DummyAllocator:
  """Docstring."""

  def get_register(self, var_name: str) -> SassRegister:
    """Docstring."""
    return SassRegister(name="R0")

  def allocate_temp(self) -> SassRegister:
    """Docstring."""
    return SassRegister(name="R1")

  def allocate_predicate(self) -> SassRegister:
    """Docstring."""
    return SassRegister(name="P0", is_predicate=True)


def test_sass_macros():
  """Docstring."""
  alloc = DummyAllocator()

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

  gen = _make_generic_expand("generic")
  assert len(gen(alloc, "g", {})) > 0
