"""Test module."""

from ml_switcheroo.core.compiler.frontends.sass.analysis import SassAnalyzer
from ml_switcheroo.core.compiler.frontends.sass.cst import SassInstruction, SassImmediate, SassRegister


def test_sass_analyzer() -> None:
  """Test element."""
  inst1: SassInstruction = SassInstruction(
    opcode="ISETP.LT.AND",
    operands=[
      SassRegister(name="P0"),
      SassRegister(name="PT"),
      SassRegister(name="R1"),
      SassImmediate(value=3),
      SassRegister(name="PT"),
    ],
  )
  inst2: SassInstruction = SassInstruction(
    opcode="ISETP.LT.AND",
    operands=[
      SassRegister(name="P0"),
      SassRegister(name="PT"),
      SassRegister(name="R2"),
      SassImmediate(value=5),
      SassRegister(name="PT"),
    ],
  )
  SassInstruction(opcode="MOV", operands=[])

  # Empty
  assert SassAnalyzer.analyze_block("Conv2d", []) == {}

  # Conv2d
  assert SassAnalyzer.analyze_block("Conv2d", [inst1, inst2]) == {"kernel_size": 5, "arg_2": 5}

  # Linear
  assert SassAnalyzer.analyze_block("Linear", [inst1, inst2]) == {"in_features": 5, "arg_0": 5}

  # Conv3d
  assert SassAnalyzer.analyze_block("Conv3d", [inst1, inst2]) == {"kernel_size": 5, "arg_2": 5}

  # AvgPool2d
  assert SassAnalyzer.analyze_block("AvgPool2d", [inst1]) == {"kernel_size": 3, "arg_2": 3}

  # MaxPool2d
  assert SassAnalyzer.analyze_block("MaxPool2d", [inst1]) == {"kernel_size": 3, "arg_2": 3}

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
    assert SassAnalyzer.analyze_block(kind, [inst1]) == {}

  # MSELoss, CrossEntropyLoss
  assert SassAnalyzer.analyze_block("MSELoss", [inst1]) == {"elements": 3, "arg_0": 3}
  assert SassAnalyzer.analyze_block("CrossEntropyLoss", [inst1]) == {"elements": 3, "arg_0": 3}

  # Mean
  assert SassAnalyzer.analyze_block("Mean", [inst1]) == {"elements": 3, "arg_0": 3}
