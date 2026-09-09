"""Tests for nvidia_sass/analysis.py."""

import typing

from ml_switcheroo.core.compiler.frontends.nvidia_sass.analysis import NvidiaSassAnalyzer
from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import (
  NvidiaSassImmediate,
  NvidiaSassInstruction,
  NvidiaSassRegister,
)


def test_nvidia_sass_analyzer_empty() -> None:
  """Verifies the behavior of nvidia_sass analyzer empty."""
  assert NvidiaSassAnalyzer.analyze_block("Conv2d", []) == {}


def test_nvidia_sass_analyzer_no_loop_limits() -> None:
  """Verifies the behavior of nvidia_sass analyzer no loop limits."""
  inst = NvidiaSassInstruction(opcode="MOV", operands=[NvidiaSassRegister(name="R0"), NvidiaSassRegister(name="R1")])
  assert NvidiaSassAnalyzer.analyze_block("Conv2d", [inst]) == {}


def test_nvidia_sass_analyzer_conv2d() -> None:
  """Verifies the behavior of nvidia_sass analyzer conv2d."""
  inst = NvidiaSassInstruction(
    opcode="ISETP.LT.AND",
    operands=[
      NvidiaSassRegister(name="P0"),
      NvidiaSassRegister(name="PT"),
      NvidiaSassRegister(name="R1"),
      NvidiaSassImmediate(value=3),
      NvidiaSassRegister(name="PT"),
    ],
  )
  res: dict[str, typing.Any] = NvidiaSassAnalyzer.analyze_block("Conv2d", [inst])
  assert res == {"kernel_size": 3, "arg_2": 3}


def test_nvidia_sass_analyzer_linear() -> None:
  """Verifies the behavior of nvidia_sass analyzer linear."""
  inst = NvidiaSassInstruction(
    opcode="ISETP.LT.AND",
    operands=[
      NvidiaSassRegister(name="P0"),
      NvidiaSassRegister(name="PT"),
      NvidiaSassRegister(name="R1"),
      NvidiaSassImmediate(value=128),
      NvidiaSassRegister(name="PT"),
    ],
  )
  res: dict[str, typing.Any] = NvidiaSassAnalyzer.analyze_block("Linear", [inst])
  assert res == {"in_features": 128, "arg_0": 128}


def test_nvidia_sass_analyzer_conv3d() -> None:
  """Verifies the behavior of nvidia_sass analyzer conv3d."""
  inst = NvidiaSassInstruction(
    opcode="ISETP.LT.AND",
    operands=[
      NvidiaSassRegister(name="P0"),
      NvidiaSassRegister(name="PT"),
      NvidiaSassRegister(name="R1"),
      NvidiaSassImmediate(value=5),
      NvidiaSassRegister(name="PT"),
    ],
  )
  res: dict[str, typing.Any] = NvidiaSassAnalyzer.analyze_block("Conv3d", [inst])
  assert res == {"kernel_size": 5, "arg_2": 5}


def test_nvidia_sass_analyzer_mean() -> None:
  """Verifies the behavior of nvidia_sass analyzer mean."""
  inst = NvidiaSassInstruction(
    opcode="ISETP.LT.AND",
    operands=[
      NvidiaSassRegister(name="P0"),
      NvidiaSassRegister(name="PT"),
      NvidiaSassRegister(name="R1"),
      NvidiaSassImmediate(value=64),
      NvidiaSassRegister(name="PT"),
    ],
  )
  res: dict[str, typing.Any] = NvidiaSassAnalyzer.analyze_block("Mean", [inst])
  assert res == {"elements": 64, "arg_0": 64}


def test_nvidia_sass_analyzer_unknown() -> None:
  """Verifies the behavior of nvidia_sass analyzer unknown."""
  inst = NvidiaSassInstruction(
    opcode="ISETP.LT.AND",
    operands=[
      NvidiaSassRegister(name="P0"),
      NvidiaSassRegister(name="PT"),
      NvidiaSassRegister(name="R1"),
      NvidiaSassImmediate(value=64),
      NvidiaSassRegister(name="PT"),
    ],
  )
  res: dict[str, typing.Any] = NvidiaSassAnalyzer.analyze_block("Unknown", [inst])
  assert res == {}


def test_analyze_block_empty_elifs() -> None:
  """Verifies the behavior of analyze block empty elifs."""
  inst = NvidiaSassInstruction(
    opcode="ISETP.LT.AND",
    operands=[
      NvidiaSassRegister(name="P0"),
      NvidiaSassRegister(name="PT"),
      NvidiaSassRegister(name="R1"),
      NvidiaSassImmediate(value=10),
      NvidiaSassRegister(name="PT"),
    ],
  )

  kinds: list[str] = [
    "AvgPool2d",
    "MaxPool2d",
    "BatchNorm2d",
    "Conv1d",
    "DepthwiseConv2d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
    "BatchNorm1d",
    "BatchNorm3d",
    "LayerNorm",
    "GroupNorm",
    "InstanceNorm2d",
    "Softmax",
    "LogSoftmax",
    "SiLU",
    "Swish",
    "ELU",
    "LeakyReLU",
    "BMM",
    "Dot",
    "SVD",
    "Solve",
    "Cholesky",
    "Sum",
    "Prod",
    "Min",
    "Max",
    "ArgMax",
    "ArgMin",
    "Any",
    "All",
    "BCEWithLogitsLoss",
    "L1Loss",
    "NLLLoss",
    "Dropout2d",
    "Dropout3d",
    "AlphaDropout",
    "AvgPool1d",
    "MaxPool1d",
    "AvgPool3d",
    "MaxPool3d",
    "AdaptiveAvgPool2d",
    "AdaptiveMaxPool2d",
    "MultiheadAttention",
    "Transformer",
    "TransformerEncoder",
    "TransformerDecoder",
    "RNN",
    "LSTM",
    "GRU",
    "LSTMCell",
    "GRUCell",
    "MSELoss",
    "CrossEntropyLoss",
    "Sigmoid",
    "Tanh",
    "GELU",
    "Dropout",
    "MatMul",
  ]
  for k in kinds:
    NvidiaSassAnalyzer.analyze_block(k, [inst])


# --- Merged from test_analysis_more.py ---


def test_nvidia_sass_analyzer_avgpool2d() -> None:
  """Verifies analyzer handles AvgPool2d limits."""
  instructions = [
    NvidiaSassInstruction(
      opcode="ISETP.LT.AND",
      operands=[
        NvidiaSassRegister(name="P0"),
        NvidiaSassRegister(name="PT"),
        NvidiaSassRegister(name="R0"),
        NvidiaSassImmediate(value=5),
        NvidiaSassRegister(name="PT"),
      ],
    )
  ]
  metadata: dict[str, typing.Any] = NvidiaSassAnalyzer.analyze_block("AvgPool2d", instructions)
  assert metadata["kernel_size"] == 5


def test_nvidia_sass_analyzer_maxpool2d() -> None:
  """Verifies analyzer handles MaxPool2d limits."""
  instructions = [
    NvidiaSassInstruction(
      opcode="ISETP.LT.AND",
      operands=[
        NvidiaSassRegister(name="P0"),
        NvidiaSassRegister(name="PT"),
        NvidiaSassRegister(name="R0"),
        NvidiaSassImmediate(value=7),
        NvidiaSassRegister(name="PT"),
      ],
    )
  ]
  metadata: dict[str, typing.Any] = NvidiaSassAnalyzer.analyze_block("MaxPool2d", instructions)
  assert metadata["kernel_size"] == 7


def test_nvidia_sass_analyzer_batchnorm2d() -> None:
  """Verifies analyzer handles BatchNorm2d safely."""
  instructions = [
    NvidiaSassInstruction(
      opcode="FADD",
      operands=[NvidiaSassRegister(name="R1"), NvidiaSassRegister(name="R2"), NvidiaSassImmediate(value=0.001)],
    )
  ]
  metadata: dict[str, typing.Any] = NvidiaSassAnalyzer.analyze_block("BatchNorm2d", instructions)
  assert len(metadata) == 0
