"""Test suite for the Analysis module."""

from ml_switcheroo.core.compiler.frontends.sass.analysis import SassAnalyzer
from ml_switcheroo.core.compiler.frontends.sass.nodes import Instruction, Register, Immediate, Predicate


def make_inst(opcode, *operands):
  """Helper to make inst."""
  return Instruction(opcode, list(operands))


def test_analyze_conv2d_kernel_size():
  """Analyzes conv2d kernel size."""
  r3 = Register("R3")
  pt = Register("PT")
  p0 = Predicate("P0")
  insts = [
    make_inst("MOV", Register("R1"), Register("RZ")),
    make_inst("ISETP.LT.AND", p0, pt, r3, Immediate(3), pt),
    make_inst("BRA", Register("L_LOOP")),
  ]
  meta = SassAnalyzer.analyze_block("Conv2d", insts)
  assert "kernel_size" in meta
  assert meta["kernel_size"] == 3
  assert meta["arg_2"] == 3


def test_analyze_linear_in_features():
  """Analyzes linear in features."""
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
  """Analyzes no loop found."""
  insts = [make_inst("FADD", Register("R0"), Register("R1"), Register("R2"))]
  meta = SassAnalyzer.analyze_block("Linear", insts)
  assert meta == {}
