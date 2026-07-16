"""Auto-generated doc."""

from ml_switcheroo.core.compiler.frontends.sass.analysis import SassAnalyzer
from ml_switcheroo.core.compiler.frontends.sass.nodes import (
  Instruction as SassInst,
  Immediate as SassImm,
  Register as SassReg,
)
from ml_switcheroo.core.compiler.frontends.rdna.analysis import RdnaAnalyzer
from ml_switcheroo.core.compiler.frontends.rdna.nodes import Instruction as RdnaInst, Immediate as RdnaImm, VGPR


def test_sass_analyzer_edge_cases():
  """Auto-generated doc."""
  # 1. Empty instructions
  assert SassAnalyzer.analyze_block("Conv2d", []) == {}

  # 2. No loop opcodes
  insts_no_loop = [SassInst("FADD", [SassReg("R0"), SassReg("R1")])]
  assert SassAnalyzer.analyze_block("Conv2d", insts_no_loop) == {}

  # 3. Loop opcode with no immediate
  insts_no_imm = [SassInst("ISETP.LT.AND", [SassReg("R0"), SassReg("R1")])]
  assert SassAnalyzer.analyze_block("Conv2d", insts_no_imm) == {}

  # 4. Unknown kind with loop opcode
  insts_loop = [SassInst("ISETP.LT.AND", [SassReg("R0"), SassImm(5)])]
  assert SassAnalyzer.analyze_block("Unknown", insts_loop) == {}

  # 5. Conv2d with multiple immediates, should take max
  insts_multi = [SassInst("ISETP.LT.AND", [SassImm(3)]), SassInst("ISETP.LT.AND", [SassImm(7)])]
  meta = SassAnalyzer.analyze_block("Conv2d", insts_multi)
  assert meta["kernel_size"] == 7
  assert meta["arg_2"] == 7


def test_rdna_analyzer_edge_cases():
  """Auto-generated doc."""
  # 1. Empty instructions
  assert RdnaAnalyzer.analyze_block("Conv2d", []) == {}

  # 2. No loop opcodes
  insts_no_loop = [RdnaInst("v_add_f32", [VGPR(0), VGPR(1)])]
  assert RdnaAnalyzer.analyze_block("Conv2d", insts_no_loop) == {}

  # 3. Loop opcode with no immediate
  insts_no_imm = [RdnaInst("s_cmp_lt_i32", [VGPR(0), VGPR(1)])]
  assert RdnaAnalyzer.analyze_block("Conv2d", insts_no_imm) == {}

  # 4. Unknown kind with loop opcode
  insts_loop = [RdnaInst("s_cmp_lt_i32", [VGPR(0), RdnaImm(5)])]
  assert RdnaAnalyzer.analyze_block("Unknown", insts_loop) == {}

  # 5. Linear with multiple immediates, should take max
  insts_multi = [RdnaInst("s_cmp_lt_i32", [RdnaImm(128)]), RdnaInst("s_cmp_lt_i32", [RdnaImm(64)])]
  meta = RdnaAnalyzer.analyze_block("Linear", insts_multi)
  assert meta["in_features"] == 128
  assert meta["arg_0"] == 128
