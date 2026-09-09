"""Test suite for NVIDIA_SASS macros integration."""

from ml_switcheroo.core.compiler.backends.nvidia_sass.macros import (
  expand_adaptivepool,
  expand_avgpool2d,
  expand_batchnorm2d,
  expand_conv1d,
  expand_conv2d,
  expand_conv3d,
  expand_convtranspose,
  expand_crossentropyloss,
  expand_depthwiseconv2d,
  expand_dropout,
  expand_flatten,
  expand_gelu,
  expand_generic_activation,
  expand_generic_dropout,
  expand_generic_linalg,
  expand_generic_loss,
  expand_generic_norm,
  expand_generic_reduction,
  expand_gru,
  expand_linear,
  expand_lstm,
  expand_maxpool2d,
  expand_mean,
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
)
from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassRegister


class DummyAllocator:
  """Docstring."""

  def __init__(self) -> None:
    """Initialize the dummy allocator."""
    self.count: int = 0

  def get_register(self, var_name: str) -> NvidiaSassRegister:
    """Get a register for a variable."""
    return NvidiaSassRegister(name=f"R_VAR_{var_name}")

  def allocate_temp(self) -> NvidiaSassRegister:
    """Allocate a temporary register."""
    self.count += 1
    return NvidiaSassRegister(name=f"R_TMP_{self.count}")


def test_macros_basic() -> None:
  """Docstring."""
  alloc = DummyAllocator()

  # Test all with basic metadata
  assert len(expand_conv2d(alloc, "test", {"k": 3})) > 0  # type: ignore[arg-type]
  assert len(expand_linear(alloc, "test", {"in_features": 128})) > 0  # type: ignore[arg-type]
  assert len(expand_mean(alloc, "test", {"elements": 128})) > 0  # type: ignore[arg-type]
  assert len(expand_relu(alloc, "test", {})) > 0  # type: ignore[arg-type]
  assert len(expand_flatten(alloc, "test", {})) > 0  # type: ignore[arg-type]
  assert len(expand_reshape(alloc, "test", {})) > 0  # type: ignore[arg-type]
  assert len(expand_conv3d(alloc, "test", {"k": 3})) > 0  # type: ignore[arg-type]
  assert len(expand_avgpool2d(alloc, "test", {"kernel_size": 3})) > 0  # type: ignore[arg-type]
  assert len(expand_maxpool2d(alloc, "test", {"kernel_size": 3})) > 0  # type: ignore[arg-type]
  assert len(expand_batchnorm2d(alloc, "test", {"eps": 1e-5})) > 0  # type: ignore[arg-type]
  assert len(expand_dropout(alloc, "test", {"p": 0.5})) > 0  # type: ignore[arg-type]
  assert len(expand_sigmoid(alloc, "test", {})) > 0  # type: ignore[arg-type]
  assert len(expand_tanh(alloc, "test", {})) > 0  # type: ignore[arg-type]
  assert len(expand_gelu(alloc, "test", {})) > 0  # type: ignore[arg-type]
  assert len(expand_mseloss(alloc, "test", {"elements": 128, "reduction": "mean"})) > 0  # type: ignore[arg-type]
  assert len(expand_crossentropyloss(alloc, "test", {"elements": 32})) > 0  # type: ignore[arg-type]
  assert len(expand_rnn(alloc, "test", {"seq_len": 10})) > 0  # type: ignore[arg-type]
  assert len(expand_lstm(alloc, "test", {"seq_len": 10})) > 0  # type: ignore[arg-type]
  assert len(expand_gru(alloc, "test", {"seq_len": 10})) > 0  # type: ignore[arg-type]
  assert len(expand_multiheadattention(alloc, "test", {})) > 0  # type: ignore[arg-type]
  assert len(expand_transformer(alloc, "test", {})) > 0  # type: ignore[arg-type]
  assert len(expand_transformerencoder(alloc, "test", {})) > 0  # type: ignore[arg-type]
  assert len(expand_transformerdecoder(alloc, "test", {})) > 0  # type: ignore[arg-type]
  assert len(expand_conv1d(alloc, "test", {"k": 3})) > 0  # type: ignore[arg-type]
  assert len(expand_depthwiseconv2d(alloc, "test", {"k": 3})) > 0  # type: ignore[arg-type]
  assert len(expand_convtranspose(alloc, "test", {})) > 0  # type: ignore[arg-type]
  assert len(expand_pool1d(alloc, "test", {})) > 0  # type: ignore[arg-type]
  assert len(expand_pool3d(alloc, "test", {})) > 0  # type: ignore[arg-type]
  assert len(expand_adaptivepool(alloc, "test", {})) > 0  # type: ignore[arg-type]
  assert len(expand_generic_norm(alloc, "test", {})) > 0  # type: ignore[arg-type]
  assert len(expand_generic_activation(alloc, "test", {})) > 0  # type: ignore[arg-type]
  assert len(expand_generic_linalg(alloc, "test", {})) > 0  # type: ignore[arg-type]
  assert len(expand_generic_reduction(alloc, "test", {})) > 0  # type: ignore[arg-type]
  assert len(expand_generic_loss(alloc, "test", {})) > 0  # type: ignore[arg-type]
  assert len(expand_generic_dropout(alloc, "test", {})) > 0  # type: ignore[arg-type]


def test_macros_branches() -> None:
  """Docstring."""
  alloc = DummyAllocator()

  # Linear with bias
  assert len(expand_linear(alloc, "test", {"in_features": 128, "bias": True})) > 0  # type: ignore[arg-type]

  # Mean with limit 0
  assert len(expand_mean(alloc, "test", {"elements": 0})) > 0  # type: ignore[arg-type]

  # AvgPool2d with kernel_size 0
  assert len(expand_avgpool2d(alloc, "test", {"kernel_size": 0})) > 0  # type: ignore[arg-type]

  # Dropout with p=1.0
  assert len(expand_dropout(alloc, "test", {"p": 1.0})) > 0  # type: ignore[arg-type]

  # MSELoss with limit 0, and non-mean reduction
  assert len(expand_mseloss(alloc, "test", {"elements": 0, "reduction": "mean"})) > 0  # type: ignore[arg-type]
  assert len(expand_mseloss(alloc, "test", {"elements": 128, "reduction": "sum"})) > 0  # type: ignore[arg-type]
