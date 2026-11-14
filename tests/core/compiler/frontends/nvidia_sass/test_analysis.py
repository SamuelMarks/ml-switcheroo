"""Unit tests for `NvidiaSassAnalyzer` (`ml_switcheroo.core.compiler.frontends.nvidia_sass.analysis`)."""

from ml_switcheroo.core.compiler.frontends.nvidia_sass.analysis import NvidiaSassAnalyzer
from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import (
  NvidiaSassImmediate,
  NvidiaSassInstruction,
  NvidiaSassLabel,
  NvidiaSassMemory,
  NvidiaSassPredicate,
  NvidiaSassRegister,
)


def test_nvidia_sass_analyzer_loop_bounds_and_op_kinds() -> None:
  """Verify `NvidiaSassAnalyzer.analyze_block` across empty, loop-free, and loop-bounded operation blocks."""
  assert NvidiaSassAnalyzer.analyze_block("Conv2d", []) == {}
  assert (
    NvidiaSassAnalyzer.analyze_block(
      "Conv2d",
      [NvidiaSassInstruction(opcode="MOV", operands=[NvidiaSassRegister(name="R0"), NvidiaSassRegister(name="R1")])],
    )
    == {}
  )

  p0 = NvidiaSassPredicate(name="P0")
  pt = NvidiaSassRegister(name="PT")
  r3 = NvidiaSassRegister(name="R3")

  conv_insts = [
    NvidiaSassInstruction(opcode="MOV", operands=[NvidiaSassRegister(name="R1"), NvidiaSassRegister(name="RZ")]),
    NvidiaSassInstruction(opcode="ISETP.LT.AND", operands=[p0, pt, r3, NvidiaSassImmediate(value=3), pt]),
    NvidiaSassInstruction(opcode="BRA", operands=[NvidiaSassLabel(name="L_LOOP")]),
  ]
  assert NvidiaSassAnalyzer.analyze_block("Conv2d", conv_insts) == {"kernel_size": 3, "arg_2": 3}

  lin_insts = [
    NvidiaSassInstruction(opcode="LDG.E.F32", operands=[NvidiaSassRegister(name="R9"), NvidiaSassMemory(base="addr")]),
    NvidiaSassInstruction(opcode="ISETP.LT.AND", operands=[p0, pt, r3, NvidiaSassImmediate(value=128), pt]),
  ]
  assert NvidiaSassAnalyzer.analyze_block("Linear", lin_insts) == {"in_features": 128, "arg_0": 128}

  bound_5 = [NvidiaSassInstruction(opcode="ISETP.LT.AND", operands=[p0, pt, r3, NvidiaSassImmediate(value=5), pt])]
  assert NvidiaSassAnalyzer.analyze_block("Conv3d", bound_5) == {"kernel_size": 5, "arg_2": 5}
  assert NvidiaSassAnalyzer.analyze_block("AvgPool2d", bound_5)["kernel_size"] == 5
  assert NvidiaSassAnalyzer.analyze_block("MaxPool2d", bound_5)["kernel_size"] == 5
  assert NvidiaSassAnalyzer.analyze_block("Mean", bound_5) == {"elements": 5, "arg_0": 5}
  assert NvidiaSassAnalyzer.analyze_block("MSELoss", bound_5)["elements"] == 5
  assert NvidiaSassAnalyzer.analyze_block("Unknown", bound_5) == {}

  other_kinds = (
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
    "CrossEntropyLoss",
    "Sigmoid",
    "Tanh",
    "GELU",
    "Dropout",
    "MatMul",
  )
  for kind in other_kinds:
    assert isinstance(NvidiaSassAnalyzer.analyze_block(kind, bound_5), dict)
