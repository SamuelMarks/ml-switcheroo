"""Test module."""

from ml_switcheroo.core.compiler.frontends.nvidia_sass.analysis import NvidiaSassAnalyzer
from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import (
  NvidiaSassImmediate,
  NvidiaSassInstruction,
  NvidiaSassRegister,
)


def test_nvidia_sass_analyzer() -> None:
  """Docstring."""
  inst1: NvidiaSassInstruction = NvidiaSassInstruction(
    opcode="ISETP.LT.AND",
    operands=[
      NvidiaSassRegister(name="P0"),
      NvidiaSassRegister(name="PT"),
      NvidiaSassRegister(name="R1"),
      NvidiaSassImmediate(value=3),
      NvidiaSassRegister(name="PT"),
    ],
  )
  inst2: NvidiaSassInstruction = NvidiaSassInstruction(
    opcode="ISETP.LT.AND",
    operands=[
      NvidiaSassRegister(name="P0"),
      NvidiaSassRegister(name="PT"),
      NvidiaSassRegister(name="R2"),
      NvidiaSassImmediate(value=5),
      NvidiaSassRegister(name="PT"),
    ],
  )
  NvidiaSassInstruction(opcode="MOV", operands=[])

  # Empty
  assert NvidiaSassAnalyzer.analyze_block("Conv2d", []) == {}

  # Conv2d
  assert NvidiaSassAnalyzer.analyze_block("Conv2d", [inst1, inst2]) == {"kernel_size": 5, "arg_2": 5}

  # Linear
  assert NvidiaSassAnalyzer.analyze_block("Linear", [inst1, inst2]) == {"in_features": 5, "arg_0": 5}

  # Conv3d
  assert NvidiaSassAnalyzer.analyze_block("Conv3d", [inst1, inst2]) == {"kernel_size": 5, "arg_2": 5}

  # AvgPool2d
  assert NvidiaSassAnalyzer.analyze_block("AvgPool2d", [inst1]) == {"kernel_size": 3, "arg_2": 3}

  # MaxPool2d
  assert NvidiaSassAnalyzer.analyze_block("MaxPool2d", [inst1]) == {"kernel_size": 3, "arg_2": 3}

  # Pass branches
  for kind in [
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
    "Sigmoid",
    "Tanh",
    "GELU",
    "Dropout",
    "MatMul",
    "UnknownKind",
  ]:
    assert NvidiaSassAnalyzer.analyze_block(kind, [inst1]) == {}

  # MSELoss, CrossEntropyLoss
  assert NvidiaSassAnalyzer.analyze_block("MSELoss", [inst1]) == {"elements": 3, "arg_0": 3}
  assert NvidiaSassAnalyzer.analyze_block("CrossEntropyLoss", [inst1]) == {"elements": 3, "arg_0": 3}

  # Mean
  assert NvidiaSassAnalyzer.analyze_block("Mean", [inst1]) == {"elements": 3, "arg_0": 3}
