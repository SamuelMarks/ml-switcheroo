"""Unit tests for SASS-based logical block analysis.

This module verifies that the SassAnalyzer correctly parses micro-architectural
SASS instructions (like ISETP conditional evaluations) and maps them back to
higher-level logical node parameters (e.g., kernel size, in_features, elements).
"""

from ml_switcheroo.core.compiler.frontends.sass.analysis import SassAnalyzer
from ml_switcheroo.core.compiler.frontends.sass.cst import SassInstruction, SassImmediate, SassRegister


def test_sass_analyzer_empty():
  """Verifies that analyzing an empty SASS block returns an empty dictionary.

  Args:
      None

  Returns:
      None
  """
  meta = SassAnalyzer.analyze_block("Conv2d", [])
  assert meta == {}


def test_sass_analyzer_no_limits():
  """Verifies that an instruction stream with no conditional limit instructions has no metadata.

  Args:
      None

  Returns:
      None
  """
  insts = [SassInstruction(opcode="FADD", operands=[SassRegister(name="R0")])]
  meta = SassAnalyzer.analyze_block("Conv2d", insts)
  assert meta == {}


def test_sass_analyzer_conv2d():
  """Verifies metadata extraction from a Conv2d SASS block.

  This checks that a comparison instruction containing an immediate value
  correctly sets the kernel_size and argument parameter for a Conv2d block.

  Args:
      None

  Returns:
      None
  """
  insts = [SassInstruction(opcode="ISETP.LT.AND", operands=[SassRegister(name="R0"), SassImmediate(value=3)])]
  meta = SassAnalyzer.analyze_block("Conv2d", insts)
  assert meta == {"kernel_size": 3, "arg_2": 3}


def test_sass_analyzer_linear():
  """Verifies metadata extraction from a Linear SASS block.

  This checks that a comparison instruction containing an immediate value
  correctly sets the in_features and argument parameter for a Linear block.

  Args:
      None

  Returns:
      None
  """
  insts = [SassInstruction(opcode="ISETP.LT.AND", operands=[SassRegister(name="R0"), SassImmediate(value=128)])]
  meta = SassAnalyzer.analyze_block("Linear", insts)
  assert meta == {"in_features": 128, "arg_0": 128}


def test_sass_analyzer_conv3d():
  """Verifies metadata extraction from a Conv3d SASS block.

  This checks that a comparison instruction containing an immediate value
  correctly sets the kernel_size and argument parameter for a Conv3d block.

  Args:
      None

  Returns:
      None
  """
  insts = [SassInstruction(opcode="ISETP.LT.AND", operands=[SassRegister(name="R0"), SassImmediate(value=5)])]
  meta = SassAnalyzer.analyze_block("Conv3d", insts)
  assert meta == {"kernel_size": 5, "arg_2": 5}


def test_sass_analyzer_mean():
  """Verifies metadata extraction from a Mean SASS block.

  This checks that a comparison instruction containing an immediate value
  correctly sets the elements count and argument parameter for a Mean block.

  Args:
      None

  Returns:
      None
  """
  insts = [SassInstruction(opcode="ISETP.LT.AND", operands=[SassRegister(name="R0"), SassImmediate(value=10)])]
  meta = SassAnalyzer.analyze_block("Mean", insts)
  assert meta == {"elements": 10, "arg_0": 10}


def test_sass_analyzer_unknown_kind():
  """Verifies that an unrecognized logical block type returns empty metadata.

  Even if the block contains valid comparison instructions, since the block type
  is unrecognized, no metadata is matched.

  Args:
      None

  Returns:
      None
  """
  insts = [SassInstruction(opcode="ISETP.LT.AND", operands=[SassRegister(name="R0"), SassImmediate(value=10)])]
  meta = SassAnalyzer.analyze_block("UnknownKind", insts)
  assert meta == {}
