from typing import List
from ml_switcheroo.compiler.frontends.sass.analysis import SassAnalyzer
from ml_switcheroo.compiler.frontends.sass.nodes import Instruction, Register, Immediate, Predicate


def make_inst(opcode, *operands):
  # Helper to build instruction
  return Instruction(opcode, list(operands))


def test_analyze_conv2d_kernel_size():
  """
  Scenario: Conv2d block with loop limit 3.
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
  """
  r8 = Register("R8")
  pt = Register("PT")
  p0 = Predicate("P0")

  insts = [
    make_inst("LDG.E.F32", Register("R9"), Register("addr")),
    make_inst("ISETP.LT.AND", p0, pt, r8, Immediate(128), pt),
  ]

  meta = SassAnalyzer.analyze_block("Linear", insts)

  assert "in_features" in meta
  assert meta["in_features"] == 128
  assert meta["arg_0"] == 128


def test_analyze_no_loop_found():
  """
  Scenario: Block with no ISETP instructions.
  """
  insts = [make_inst("FADD", Register("R0"), Register("R1"), Register("R2"))]
  meta = SassAnalyzer.analyze_block("Linear", insts)
  assert meta == {}
