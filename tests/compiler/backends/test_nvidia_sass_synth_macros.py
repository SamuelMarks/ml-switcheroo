"""Test suite for nvidia sass synth macros."""

from ml_switcheroo.core.compiler.backends.nvidia_sass.macros import (
  expand_adam,
  expand_adaptivepool,
  expand_avgpool2d,
  expand_batchnorm2d,
  expand_conv1d,
  expand_conv2d,
  expand_conv3d,
  expand_conv_general_dilated,
  expand_convtranspose,
  expand_crossentropyloss,
  expand_depthwiseconv2d,
  expand_dropout,
  expand_flatten,
  expand_gelu,
  expand_gru,
  expand_l,
  expand_linear,
  expand_lstm,
  expand_maxpool2d,
  expand_mseloss,
  expand_multiheadattention,
  expand_pool1d,
  expand_pool3d,
  expand_relu,
  expand_reshape,
  expand_rnn,
  expand_sigmoid,
  expand_tanh,
  expand_transformer,
  expand_transformerdecoder,
  expand_transformerencoder,
  expand_transpose,
  expand_variable,
)
from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassRegister


class MockAllocator:
  """Test helper class MockAllocator."""

  def get_register(self, var_name: str) -> NvidiaSassRegister:  # noqa: F811
    """Execute get register helper logic."""
    return NvidiaSassRegister(var_name)

  def allocate_temp(self) -> NvidiaSassRegister:  # noqa: F811
    """Execute allocate temp helper logic."""
    return NvidiaSassRegister("TEMP")

  def free_register(self, var_name: str) -> None:
    """Execute free register helper logic."""
    pass


def test_all_macros() -> None:
  """Verify all macros behavior."""
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
