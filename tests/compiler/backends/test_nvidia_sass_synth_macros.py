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
    """Execute get register helper logic.

    Args:
        var_name: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return NvidiaSassRegister(name=var_name)

  def allocate_temp(self) -> NvidiaSassRegister:  # noqa: F811
    """Execute allocate temp helper logic.

    Returns:
        Computed test result or fixture value.
    """
    return NvidiaSassRegister(name="TEMP")

  def free_register(self, var_name: str) -> None:
    """Execute free register helper logic.

    Args:
        var_name: Test argument or fixture.
    """
    pass


def test_all_macros() -> None:
  """Verify all macros behavior."""
  from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import (
    NvidiaSassComment,
    NvidiaSassInstruction,
    NvidiaSassNode,
  )

  alloc = MockAllocator()
  macros_and_args = [
    (expand_conv2d, {"k": 3}),
    (expand_linear, {"in_features": 3}),
    (expand_relu, {}),
    (expand_flatten, {}),
    (expand_reshape, {}),
    (expand_conv3d, {"kernel_size": 3}),
    (expand_dropout, {}),
    (expand_variable, {}),
    (expand_transpose, {}),
    (expand_conv_general_dilated, {}),
    (expand_adam, {}),
    (expand_l, {}),
    (expand_conv1d, {"kernel_size": 3}),
    (expand_avgpool2d, {"kernel_size": 3}),
    (expand_maxpool2d, {"kernel_size": 3}),
    (expand_batchnorm2d, {}),
    (expand_sigmoid, {}),
    (expand_tanh, {}),
    (expand_gelu, {}),
    (expand_mseloss, {"elements": 3}),
    (expand_crossentropyloss, {"elements": 3}),
    (expand_rnn, {}),
    (expand_lstm, {}),
    (expand_gru, {}),
    (expand_multiheadattention, {}),
    (expand_transformer, {}),
    (expand_transformerencoder, {}),
    (expand_transformerdecoder, {}),
    (expand_depthwiseconv2d, {"kernel_size": 3}),
    (expand_convtranspose, {"kernel_size": 3}),
    (expand_pool1d, {"kernel_size": 3}),
    (expand_pool3d, {"kernel_size": 3}),
    (expand_adaptivepool, {"kernel_size": 3}),
  ]
  for fn, meta in macros_and_args:
    nodes: list[NvidiaSassNode] = fn(alloc, "n1", meta)
    assert len(nodes) > 0
    assert all(isinstance(n, NvidiaSassNode) for n in nodes)
    assert any(isinstance(n, (NvidiaSassInstruction, NvidiaSassComment)) for n in nodes)
