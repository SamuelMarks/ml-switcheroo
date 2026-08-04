"""Test suite for extended SassAnalyzer edge cases and new macros."""

from ml_switcheroo.core.compiler.frontends.sass.analysis import SassAnalyzer
from ml_switcheroo.core.compiler.frontends.sass.cst import SassInstruction, SassImmediate, SassRegister


def test_sass_analyzer_avgpool2d():
  """Verifies analyzer handles AvgPool2d limits."""
  instructions = [
    SassInstruction(
      opcode="ISETP.LT.AND",
      operands=[
        SassRegister(name="P0"),
        SassRegister(name="PT"),
        SassRegister(name="R0"),
        SassImmediate(value=5),
        SassRegister(name="PT"),
      ],
    )
  ]
  metadata = SassAnalyzer.analyze_block("AvgPool2d", instructions)
  assert metadata["kernel_size"] == 5


def test_sass_analyzer_maxpool2d():
  """Verifies analyzer handles MaxPool2d limits."""
  instructions = [
    SassInstruction(
      opcode="ISETP.LT.AND",
      operands=[
        SassRegister(name="P0"),
        SassRegister(name="PT"),
        SassRegister(name="R0"),
        SassImmediate(value=7),
        SassRegister(name="PT"),
      ],
    )
  ]
  metadata = SassAnalyzer.analyze_block("MaxPool2d", instructions)
  assert metadata["kernel_size"] == 7


def test_sass_analyzer_batchnorm2d():
  """Verifies analyzer handles BatchNorm2d safely."""
  instructions = [
    SassInstruction(
      opcode="FADD", operands=[SassRegister(name="R1"), SassRegister(name="R2"), SassImmediate(value=0.001)]
    )
  ]
  metadata = SassAnalyzer.analyze_block("BatchNorm2d", instructions)
  assert len(metadata) == 0
