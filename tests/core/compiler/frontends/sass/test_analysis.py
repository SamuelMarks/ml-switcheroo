"""Unit tests for the SASS instruction analyzer frontend.

This module validates that the heuristics in `SassAnalyzer` can correctly
reverse-engineer high-level operation parameters (such as kernel sizes,
feature counts, and element counts) from lists of low-level SASS instructions
under various scenarios (empty inputs, unknown ops, or expected workloads like
Conv2d, Linear, Conv3d, and Mean).

Args:
    None

Returns:
    None
"""

from ml_switcheroo.core.compiler.frontends.sass.analysis import SassAnalyzer
from ml_switcheroo.core.compiler.frontends.sass.cst import SassInstruction, SassImmediate, SassRegister


def test_sass_analyzer_empty():
  """Verifies that an empty instruction sequence yields an empty metadata dictionary.

  This test checks the fallback logic of the analyzer when no instructions
  are supplied to the `analyze_block` function.

  Args:
      None

  Returns:
      None
  """
  assert SassAnalyzer.analyze_block("Conv2d", []) == {}


def test_sass_analyzer_no_loop_limits():
  """Verifies that instructions without loop bounds yield empty metadata.

  This test feeds instructions (e.g., a "MOV" opcode) that do not contain loop-bound
  checks like "ISETP.LT.AND" to ensure that the heuristic returns an empty dict.

  Args:
      None

  Returns:
      None
  """
  inst = SassInstruction(opcode="MOV", operands=[SassRegister(name="R0"), SassRegister(name="R1")])
  assert SassAnalyzer.analyze_block("Conv2d", [inst]) == {}


def test_sass_analyzer_conv2d():
  """Verifies metadata extraction for 2D convolutions (Conv2d).

  This test models the loop bounds logic of a 2D convolution kernel, confirming
  that `SassAnalyzer` extracts the loop limit as `kernel_size` and `arg_2`.

  Args:
      None

  Returns:
      None
  """
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
  """Verifies metadata extraction for linear layers (Linear).

  This test models a linear dot-product loop bound check, confirming that
  `SassAnalyzer` extracts the input feature dimension as `in_features` and `arg_0`.

  Args:
      None

  Returns:
      None
  """
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
  """Verifies metadata extraction for 3D convolutions (Conv3d).

  This test models loop bounds in a 3D convolution kernel, verifying that
  `SassAnalyzer` extracts the loop limit as `kernel_size` and `arg_2`.

  Args:
      None

  Returns:
      None
  """
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
  """Verifies metadata extraction for mean reduction operations (Mean).

  This test checks reduction loop bounds, confirming that `SassAnalyzer`
  correctly extracts the element count as `elements` and `arg_0`.

  Args:
      None

  Returns:
      None
  """
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
  """Verifies that unknown operation kinds yield empty metadata.

  This test presents a valid sequence of instructions containing loop bounds but
  under an unrecognized operation type, confirming that an empty dict is returned.

  Args:
      None

  Returns:
      None
  """
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
