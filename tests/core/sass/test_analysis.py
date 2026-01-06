"""
Tests for SASS Instruction Analyzer.

Verifies that parameter extraction heuristics work on standard SASS logic blocks.
"""

from typing import List
from ml_switcheroo.core.sass.analysis import SassAnalyzer
from ml_switcheroo.core.sass.nodes import Instruction, Register, Immediate, Predicate


def make_inst(opcode, *operands):
  # Helper to build instruction
  return Instruction(opcode, list(operands))


def test_analyze_conv2d_kernel_size():
  """
  Scenario: Conv2d block with loop limit 3.
  Instructions: ... ISETP.LT.AND P0, PT, R3, 3, PT ...
  Expectation: kernel_size = 3.
  """
  # Mock sequence: loop check
  r3 = Register("R3")
  pt = Register("PT")
  p0 = Predicate("P0")

  insts = [
    make_inst("MOV", Register("R1"), Register("RZ")),
    # Compare R3 < 3
    make_inst("ISETP.LT.AND", p0, pt, r3, Immediate(3), pt),
    make_inst("BRA", Register("L_LOOP")),
  ]

  meta = SassAnalyzer.analyze_block("Conv2d", insts)

  assert "kernel_size" in meta
  assert meta["kernel_size"] == 3
  assert meta["arg_2"] == 3  # Positional fallback


def test_analyze_linear_in_features():
  """
  Scenario: Linear block with loop limit 128.
  Instructions: ... ISETP.LT.AND P0, PT, R8, 128, PT ...
  Expectation: in_features = 128.
  """
  r8 = Register("R8")
  pt = Register("PT")
  p0 = Predicate("P0")

  insts = [
    make_inst("LDG.E.F32", Register("R9"), Register("addr")),
    # Compare R8 < 128
    make_inst("ISETP.LT.AND", p0, pt, r8, Immediate(128), pt),
  ]

  meta = SassAnalyzer.analyze_block("Linear", insts)

  assert "in_features" in meta
  assert meta["in_features"] == 128
  assert meta["arg_0"] == 128


def test_analyze_no_loop_found():
  """
  Scenario: Block with no ISETP instructions (e.g. unrolled, or different logic).
  Expectation: Empty metadata.
  """
  insts = [make_inst("FADD", Register("R0"), Register("R1"), Register("R2"))]
  meta = SassAnalyzer.analyze_block("Linear", insts)
  assert meta == {}
