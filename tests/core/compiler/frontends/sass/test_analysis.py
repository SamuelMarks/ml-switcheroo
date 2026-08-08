"""Tests for sass/analysis.py."""

from ml_switcheroo.core.compiler.frontends.sass.cst import SassImmediate, SassInstruction, SassRegister
from ml_switcheroo.core.compiler.frontends.sass.analysis import SassAnalyzer


def test_sass_analyzer_empty():
  """Verifies the behavior of sass analyzer empty."""
  assert SassAnalyzer.analyze_block("Conv2d", []) == {}


def test_sass_analyzer_no_loop_limits():
  """Verifies the behavior of sass analyzer no loop limits."""
  inst = SassInstruction(opcode="MOV", operands=[SassRegister(name="R0"), SassRegister(name="R1")])
  assert SassAnalyzer.analyze_block("Conv2d", [inst]) == {}


def test_sass_analyzer_conv2d():
  """Verifies the behavior of sass analyzer conv2d."""
  inst = SassInstruction(
    opcode="ISETP.LT.AND",
    operands=[
      SassRegister(name="P0"),
      SassRegister(name="PT"),
      SassRegister(name="R1"),
      SassImmediate(value=3),
      SassRegister(name="PT"),
    ],
  )
  res = SassAnalyzer.analyze_block("Conv2d", [inst])
  assert res == {"kernel_size": 3, "arg_2": 3}


def test_sass_analyzer_linear():
  """Verifies the behavior of sass analyzer linear."""
  inst = SassInstruction(
    opcode="ISETP.LT.AND",
    operands=[
      SassRegister(name="P0"),
      SassRegister(name="PT"),
      SassRegister(name="R1"),
      SassImmediate(value=128),
      SassRegister(name="PT"),
    ],
  )
  res = SassAnalyzer.analyze_block("Linear", [inst])
  assert res == {"in_features": 128, "arg_0": 128}


def test_sass_analyzer_conv3d():
  """Verifies the behavior of sass analyzer conv3d."""
  inst = SassInstruction(
    opcode="ISETP.LT.AND",
    operands=[
      SassRegister(name="P0"),
      SassRegister(name="PT"),
      SassRegister(name="R1"),
      SassImmediate(value=5),
      SassRegister(name="PT"),
    ],
  )
  res = SassAnalyzer.analyze_block("Conv3d", [inst])
  assert res == {"kernel_size": 5, "arg_2": 5}


def test_sass_analyzer_mean():
  """Verifies the behavior of sass analyzer mean."""
  inst = SassInstruction(
    opcode="ISETP.LT.AND",
    operands=[
      SassRegister(name="P0"),
      SassRegister(name="PT"),
      SassRegister(name="R1"),
      SassImmediate(value=64),
      SassRegister(name="PT"),
    ],
  )
  res = SassAnalyzer.analyze_block("Mean", [inst])
  assert res == {"elements": 64, "arg_0": 64}


def test_sass_analyzer_unknown():
  """Verifies the behavior of sass analyzer unknown."""
  inst = SassInstruction(
    opcode="ISETP.LT.AND",
    operands=[
      SassRegister(name="P0"),
      SassRegister(name="PT"),
      SassRegister(name="R1"),
      SassImmediate(value=64),
      SassRegister(name="PT"),
    ],
  )
  res = SassAnalyzer.analyze_block("Unknown", [inst])
  assert res == {}


def test_analyze_block_empty_elifs():
  """Verifies the behavior of analyze block empty elifs."""
  inst = SassInstruction(
    opcode="ISETP.LT.AND",
    operands=[
      SassRegister(name="P0"),
      SassRegister(name="PT"),
      SassRegister(name="R1"),
      SassImmediate(value=10),
      SassRegister(name="PT"),
    ],
  )

  kinds = [
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
    SassAnalyzer.analyze_block(k, [inst])
