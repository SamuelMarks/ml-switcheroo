"""Test suite for the Analyzer Edge Cases module."""

from ml_switcheroo.core.compiler.frontends.sass.analysis import SassAnalyzer
from ml_switcheroo.core.compiler.frontends.sass.cst import (
  SassInstruction as SassInst,
  SassImmediate as SassImm,
  SassRegister as SassReg,
)
from ml_switcheroo.core.compiler.frontends.rdna.analysis import RdnaAnalyzer
from ml_switcheroo.core.compiler.frontends.rdna.cst import (
  RdnaInstruction as RdnaInst,
  RdnaImmediate as RdnaImm,
  RdnaVGPR as VGPR,
)


def test_sass_analyzer_edge_cases():
  """Verifies the behavior of SASS analyzer edge cases."""
  assert SassAnalyzer.analyze_block("Conv2d", []) == {}
  insts_no_loop = [SassInst(opcode="FADD", operands=[SassReg(name="R0"), SassReg(name="R1")])]
  assert SassAnalyzer.analyze_block("Conv2d", insts_no_loop) == {}
  insts_no_imm = [SassInst(opcode="ISETP.LT.AND", operands=[SassReg(name="R0"), SassReg(name="R1")])]
  assert SassAnalyzer.analyze_block("Conv2d", insts_no_imm) == {}
  insts_loop = [SassInst(opcode="ISETP.LT.AND", operands=[SassReg(name="R0"), SassImm(value=5)])]
  assert SassAnalyzer.analyze_block("Unknown", insts_loop) == {}
  insts_multi = [
    SassInst(opcode="ISETP.LT.AND", operands=[SassImm(value=3)]),
    SassInst(opcode="ISETP.LT.AND", operands=[SassImm(value=7)]),
  ]
  meta = SassAnalyzer.analyze_block("Conv2d", insts_multi)
  assert meta["kernel_size"] == 7
  assert meta["arg_2"] == 7


def test_rdna_analyzer_edge_cases():
  """Verifies the behavior of RDNA analyzer edge cases."""
  assert RdnaAnalyzer.analyze_block("Conv2d", []) == {}
  insts_no_loop = [RdnaInst(opcode="v_add_f32", operands=[VGPR(index=0), VGPR(index=1)])]
  assert RdnaAnalyzer.analyze_block("Conv2d", insts_no_loop) == {}
  insts_no_imm = [RdnaInst(opcode="s_cmp_lt_i32", operands=[VGPR(index=0), VGPR(index=1)])]
  assert RdnaAnalyzer.analyze_block("Conv2d", insts_no_imm) == {}
  insts_loop = [RdnaInst(opcode="s_cmp_lt_i32", operands=[VGPR(index=0), RdnaImm(value=5)])]
  assert RdnaAnalyzer.analyze_block("Unknown", insts_loop) == {}
  insts_multi = [
    RdnaInst(opcode="s_cmp_lt_i32", operands=[RdnaImm(value=128)]),
    RdnaInst(opcode="s_cmp_lt_i32", operands=[RdnaImm(value=64)]),
  ]
  meta = RdnaAnalyzer.analyze_block("Linear", insts_multi)
  assert meta["in_features"] == 128
  assert meta["arg_0"] == 128
