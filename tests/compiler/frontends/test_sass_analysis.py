"""Unit tests for SASS-based logical block analysis.

This module verifies that the SassAnalyzer correctly parses micro-architectural
SASS instructions (like ISETP conditional evaluations) and maps them back to
higher-level logical node parameters (e.g., kernel size, in_features, elements).
"""

import typing
from ml_switcheroo.core.compiler.frontends.sass.analysis import SassAnalyzer
from ml_switcheroo.core.compiler.frontends.sass.cst import SassInstruction, SassImmediate, SassRegister


def test_sass_analyzer_empty() -> None:
  """Verifies that analyzing an empty SASS block returns an empty dictionary.

  Args:
      None

  Returns:
      None
  """
  meta: dict[str, typing.Any] = SassAnalyzer.analyze_block("Conv2d", [])
  assert meta == {}


def test_sass_analyzer_no_limits() -> None:
  """Verifies that an instruction stream with no conditional limit instructions has no metadata.

  Args:
      None

  Returns:
      None
  """
  insts = [SassInstruction(opcode="FADD", operands=[SassRegister(name="R0")])]
  meta: dict[str, typing.Any] = SassAnalyzer.analyze_block("Conv2d", insts)
  assert meta == {}


def test_sass_analyzer_conv2d() -> None:
  """Verifies metadata extraction from a Conv2d SASS block.

  This checks that a comparison instruction containing an immediate value
  correctly sets the kernel_size and argument parameter for a Conv2d block.

  Args:
      None

  Returns:
      None
  """
  insts = [SassInstruction(opcode="ISETP.LT.AND", operands=[SassRegister(name="R0"), SassImmediate(value=3)])]
  meta: dict[str, typing.Any] = SassAnalyzer.analyze_block("Conv2d", insts)
  assert meta == {"kernel_size": 3, "arg_2": 3}


def test_sass_analyzer_linear() -> None:
  """Verifies metadata extraction from a Linear SASS block.

  This checks that a comparison instruction containing an immediate value
  correctly sets the in_features and argument parameter for a Linear block.

  Args:
      None

  Returns:
      None
  """
  insts = [SassInstruction(opcode="ISETP.LT.AND", operands=[SassRegister(name="R0"), SassImmediate(value=128)])]
  meta: dict[str, typing.Any] = SassAnalyzer.analyze_block("Linear", insts)
  assert meta == {"in_features": 128, "arg_0": 128}


def test_sass_analyzer_conv3d() -> None:
  """Verifies metadata extraction from a Conv3d SASS block.

  This checks that a comparison instruction containing an immediate value
  correctly sets the kernel_size and argument parameter for a Conv3d block.

  Args:
      None

  Returns:
      None
  """
  insts = [SassInstruction(opcode="ISETP.LT.AND", operands=[SassRegister(name="R0"), SassImmediate(value=5)])]
  meta: dict[str, typing.Any] = SassAnalyzer.analyze_block("Conv3d", insts)
  assert meta == {"kernel_size": 5, "arg_2": 5}


def test_sass_analyzer_mean() -> None:
  """Verifies metadata extraction from a Mean SASS block.

  This checks that a comparison instruction containing an immediate value
  correctly sets the elements count and argument parameter for a Mean block.

  Args:
      None

  Returns:
      None
  """
  insts = [SassInstruction(opcode="ISETP.LT.AND", operands=[SassRegister(name="R0"), SassImmediate(value=10)])]
  meta: dict[str, typing.Any] = SassAnalyzer.analyze_block("Mean", insts)
  assert meta == {"elements": 10, "arg_0": 10}


def test_sass_analyzer_unknown_kind() -> None:
  """Verifies that an unrecognized logical block type returns empty metadata.

  Even if the block contains valid comparison instructions, since the block type
  is unrecognized, no metadata is matched.

  Args:
      None

  Returns:
      None
  """
  insts = [SassInstruction(opcode="ISETP.LT.AND", operands=[SassRegister(name="R0"), SassImmediate(value=10)])]
  meta: dict[str, typing.Any] = SassAnalyzer.analyze_block("UnknownKind", insts)
  assert meta == {}


def test_sass_analysis_all_pass_blocks() -> None:
  """Test sass analysis all pass blocks."""
  from ml_switcheroo.core.compiler.frontends.sass.analysis import SassAnalyzer
  from ml_switcheroo.core.compiler.frontends.sass.cst import SassInstruction, SassImmediate

  _analyzer = SassAnalyzer()

  inst = SassInstruction(opcode="ISETP.LT.AND", operands=[SassImmediate(value=10)])

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
    res: dict[str, typing.Any] = SassAnalyzer.analyze_block(kind, [inst])
    if kind in ["Conv3d", "AvgPool2d", "MSELoss"]:
      assert "kernel_size" in res or "elements" in res


def test_sass_analysis_linear_no_loop_limits() -> None:
  # Hit 52->116 (Linear with no limits)
  """Test sass analysis linear no loop limits."""
  from ml_switcheroo.core.compiler.frontends.sass.analysis import SassAnalyzer

  res: dict[str, typing.Any] = SassAnalyzer.analyze_block("Linear", [])
  assert res == {}


def test_sass_analyzer_linear_no_loop_limits() -> None:
  """Test SassAnalyzer for Linear kind with empty loop limits."""
  from ml_switcheroo.core.compiler.frontends.sass.analysis import SassAnalyzer

  analyzer = SassAnalyzer()
  metadata: dict[str, typing.Any] = analyzer.analyze_block("Linear", [])
  assert "in_features" not in metadata
