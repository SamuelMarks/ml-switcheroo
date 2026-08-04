"""Test suite for the Analysis module."""

from ml_switcheroo.core.compiler.frontends.sass.analysis import SassAnalyzer
from ml_switcheroo.core.compiler.frontends.sass.cst import SassInstruction, SassRegister, SassImmediate, SassPredicate


def make_inst(opcode, *operands):
  """Helper to make inst."""
  return SassInstruction(opcode=opcode, operands=list(operands))


def test_analyze_conv2d_kernel_size():
  """Analyzes conv2d kernel size."""
  r3 = SassRegister(name="R3")
  pt = SassRegister(name="PT")
  p0 = SassPredicate(name="P0")
  insts = [
    make_inst("MOV", SassRegister(name="R1"), SassRegister(name="RZ")),
    make_inst("ISETP.LT.AND", p0, pt, r3, SassImmediate(value=3), pt),
    make_inst("BRA", SassRegister(name="L_LOOP")),
  ]
  meta = SassAnalyzer.analyze_block("Conv2d", insts)
  assert "kernel_size" in meta
  assert meta["kernel_size"] == 3
  assert meta["arg_2"] == 3


def test_analyze_linear_in_features():
  """Analyzes linear in features."""
  r8 = SassRegister(name="R8")
  pt = SassRegister(name="PT")
  p0 = SassPredicate(name="P0")
  insts = [
    make_inst("LDG.E.F32", SassRegister(name="R9"), SassRegister(name="addr")),
    make_inst("ISETP.LT.AND", p0, pt, r8, SassImmediate(value=128), pt),
  ]
  meta = SassAnalyzer.analyze_block("Linear", insts)
  assert "in_features" in meta
  assert meta["in_features"] == 128
  assert meta["arg_0"] == 128


def test_analyze_no_loop_found():
  """Analyzes no loop found."""
  insts = [make_inst("FADD", SassRegister(name="R0"), SassRegister(name="R1"), SassRegister(name="R2"))]
  meta = SassAnalyzer.analyze_block("Linear", insts)
  assert meta == {}
