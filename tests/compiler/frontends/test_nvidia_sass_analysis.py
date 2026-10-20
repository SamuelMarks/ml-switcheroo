"""Test suite for nvidia sass analysis."""

from ml_switcheroo.core.compiler.frontends.nvidia_sass.analysis import NvidiaSassAnalyzer
from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import (
  NvidiaSassImmediate,
  NvidiaSassInstruction,
  NvidiaSassRegister,
)


def test_nvidia_sass_analyzer_empty() -> None:
  """Verifies that analyzing an empty NVIDIA_SASS block returns an empty dictionary.

  Args:
      None

  Returns:
      None
  """
  meta = NvidiaSassAnalyzer.analyze_block("Conv2d", [])
  assert meta == {}


def test_nvidia_sass_analyzer_no_limits() -> None:
  """Verifies that an instruction stream with no conditional limit instructions has no metadata.

  Args:
      None

  Returns:
      None
  """
  insts = [NvidiaSassInstruction(opcode="FADD", operands=[NvidiaSassRegister(name="R0")])]
  meta = NvidiaSassAnalyzer.analyze_block("Conv2d", insts)
  assert meta == {}


def test_nvidia_sass_analyzer_conv2d() -> None:
  """Verifies metadata extraction from a Conv2d NVIDIA_SASS block.

  This checks that a comparison instruction containing an immediate value
  correctly sets the kernel_size and argument parameter for a Conv2d block.

  Args:
      None

  Returns:
      None
  """
  insts = [
    NvidiaSassInstruction(opcode="ISETP.LT.AND", operands=[NvidiaSassRegister(name="R0"), NvidiaSassImmediate(value=3)])
  ]
  meta = NvidiaSassAnalyzer.analyze_block("Conv2d", insts)
  assert meta == {"kernel_size": 3, "arg_2": 3}


def test_nvidia_sass_analyzer_linear() -> None:
  """Verifies metadata extraction from a Linear NVIDIA_SASS block.

  This checks that a comparison instruction containing an immediate value
  correctly sets the in_features and argument parameter for a Linear block.

  Args:
      None

  Returns:
      None
  """
  insts = [
    NvidiaSassInstruction(opcode="ISETP.LT.AND", operands=[NvidiaSassRegister(name="R0"), NvidiaSassImmediate(value=128)])
  ]
  meta = NvidiaSassAnalyzer.analyze_block("Linear", insts)
  assert meta == {"in_features": 128, "arg_0": 128}


def test_nvidia_sass_analyzer_conv3d() -> None:
  """Verifies metadata extraction from a Conv3d NVIDIA_SASS block.

  This checks that a comparison instruction containing an immediate value
  correctly sets the kernel_size and argument parameter for a Conv3d block.

  Args:
      None

  Returns:
      None
  """
  insts = [
    NvidiaSassInstruction(opcode="ISETP.LT.AND", operands=[NvidiaSassRegister(name="R0"), NvidiaSassImmediate(value=5)])
  ]
  meta = NvidiaSassAnalyzer.analyze_block("Conv3d", insts)
  assert meta == {"kernel_size": 5, "arg_2": 5}


def test_nvidia_sass_analyzer_mean() -> None:
  """Verifies metadata extraction from a Mean NVIDIA_SASS block.

  This checks that a comparison instruction containing an immediate value
  correctly sets the elements count and argument parameter for a Mean block.

  Args:
      None

  Returns:
      None
  """
  insts = [
    NvidiaSassInstruction(opcode="ISETP.LT.AND", operands=[NvidiaSassRegister(name="R0"), NvidiaSassImmediate(value=10)])
  ]
  meta = NvidiaSassAnalyzer.analyze_block("Mean", insts)
  assert meta == {"elements": 10, "arg_0": 10}


def test_nvidia_sass_analyzer_unknown_kind() -> None:
  """Verifies that an unrecognized logical block type returns empty metadata.

  Even if the block contains valid comparison instructions, since the block type
  is unrecognized, no metadata is matched.

  Args:
      None

  Returns:
      None
  """
  insts = [
    NvidiaSassInstruction(opcode="ISETP.LT.AND", operands=[NvidiaSassRegister(name="R0"), NvidiaSassImmediate(value=10)])
  ]
  meta = NvidiaSassAnalyzer.analyze_block("UnknownKind", insts)
  assert meta == {}


def test_nvidia_sass_analysis_all_pass_blocks() -> None:
  """Verify nvidia sass analysis all pass blocks behavior."""
  from ml_switcheroo.core.compiler.frontends.nvidia_sass.analysis import NvidiaSassAnalyzer
  from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassImmediate, NvidiaSassInstruction

  _analyzer = NvidiaSassAnalyzer()

  inst = NvidiaSassInstruction(opcode="ISETP.LT.AND", operands=[NvidiaSassImmediate(value=10)])

  kinds: list[str] = [
    "Conv3d",
    "AvgPool2d",
    "BatchNorm2d",
    "Conv1d",
    "BatchNorm1d",
    "Softmax",
    "BMM",
    "Sum",
    "BCEWithLogitsLoss",
    "Dropout2d",
    "AvgPool1d",
    "MultiheadAttention",
    "RNN",
    "MSELoss",
    "Sigmoid",
    "Dropout",
  ]

  for kind in kinds:
    res = NvidiaSassAnalyzer.analyze_block(kind, [inst])
    if kind in ["Conv3d", "AvgPool2d", "MSELoss"]:
      assert "kernel_size" in res or "elements" in res


def test_nvidia_sass_analysis_linear_no_loop_limits() -> None:
  """Verify nvidia sass analysis linear no loop limits behavior."""
  # Hit 52->116 (Linear with no limits)
  from ml_switcheroo.core.compiler.frontends.nvidia_sass.analysis import NvidiaSassAnalyzer

  res = NvidiaSassAnalyzer.analyze_block("Linear", [])
  assert res == {}


def test_nvidia_sass_analyzer_linear_no_loop_limits() -> None:
  """Verify nvidia sass analyzer linear no loop limits behavior."""
  from ml_switcheroo.core.compiler.frontends.nvidia_sass.analysis import NvidiaSassAnalyzer

  analyzer = NvidiaSassAnalyzer()
  metadata = analyzer.analyze_block("Linear", [])
  assert "in_features" not in metadata
