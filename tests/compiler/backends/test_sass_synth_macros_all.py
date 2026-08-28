"""Docstring."""

from ml_switcheroo.core.compiler.backends.sass.macros import (
  expand_conv2d,
  expand_linear,
  expand_relu,
  expand_flatten,
  expand_reshape,
  expand_conv3d,
  expand_avgpool2d,
  expand_maxpool2d,
  expand_batchnorm2d,
  expand_dropout,
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
)
from ml_switcheroo.core.compiler.frontends.sass.cst import SassRegister


class MockAllocator:
  """Docstring."""

  def get_register(self, var_name: str) -> SassRegister:
    """Docstring."""
    return SassRegister(var_name)

  def allocate_temp(self) -> SassRegister:
    """Docstring."""
    return SassRegister("TEMP")

  def free_register(self, var_name: str) -> None:
    """Docstring."""
    pass


def test_all_macros() -> None:
  """Docstring."""
  alloc = MockAllocator()
  assert len(expand_conv2d(alloc, "n1", {"k": 3})) > 0  # type: ignore
  assert len(expand_linear(alloc, "n1", {"in_features": 3})) > 0  # type: ignore
  assert len(expand_relu(alloc, "n1", {})) > 0  # type: ignore
  assert len(expand_flatten(alloc, "n1", {})) > 0  # type: ignore
  assert len(expand_reshape(alloc, "n1", {})) > 0  # type: ignore
  assert len(expand_conv3d(alloc, "n1", {"kernel_size": 3})) > 0  # type: ignore
  assert len(expand_dropout(alloc, "n1", {})) > 0  # type: ignore
  assert len(expand_variable(alloc, "n1", {})) > 0  # type: ignore
  assert len(expand_transpose(alloc, "n1", {})) > 0  # type: ignore
  assert len(expand_conv_general_dilated(alloc, "n1", {})) > 0  # type: ignore
  assert len(expand_adam(alloc, "n1", {})) > 0  # type: ignore
  assert len(expand_l(alloc, "n1", {})) > 0  # type: ignore

  assert len(expand_conv1d(alloc, "n1", {"kernel_size": 3})) > 0  # type: ignore
  assert len(expand_avgpool2d(alloc, "n1", {"kernel_size": 3})) > 0  # type: ignore
  assert len(expand_maxpool2d(alloc, "n1", {"kernel_size": 3})) > 0  # type: ignore
  assert len(expand_batchnorm2d(alloc, "n1", {})) > 0  # type: ignore
  assert len(expand_sigmoid(alloc, "n1", {})) > 0  # type: ignore
  assert len(expand_tanh(alloc, "n1", {})) > 0  # type: ignore
  assert len(expand_gelu(alloc, "n1", {})) > 0  # type: ignore
  assert len(expand_mseloss(alloc, "n1", {"elements": 3})) > 0  # type: ignore
  assert len(expand_crossentropyloss(alloc, "n1", {"elements": 3})) > 0  # type: ignore

  # Extra
  assert len(expand_rnn(alloc, "n1", {})) > 0  # type: ignore
  assert len(expand_lstm(alloc, "n1", {})) > 0  # type: ignore
  assert len(expand_gru(alloc, "n1", {})) > 0  # type: ignore
  assert len(expand_multiheadattention(alloc, "n1", {})) > 0  # type: ignore
  assert len(expand_transformer(alloc, "n1", {})) > 0  # type: ignore
  assert len(expand_transformerencoder(alloc, "n1", {})) > 0  # type: ignore
  assert len(expand_transformerdecoder(alloc, "n1", {})) > 0  # type: ignore
  assert len(expand_depthwiseconv2d(alloc, "n1", {"kernel_size": 3})) > 0  # type: ignore
  assert len(expand_convtranspose(alloc, "n1", {"kernel_size": 3})) > 0  # type: ignore
  assert len(expand_pool1d(alloc, "n1", {"kernel_size": 3})) > 0  # type: ignore
  assert len(expand_pool3d(alloc, "n1", {"kernel_size": 3})) > 0  # type: ignore
  assert len(expand_adaptivepool(alloc, "n1", {"kernel_size": 3})) > 0  # type: ignore
