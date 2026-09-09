"""Test suite for the Analyzer Edge Cases module."""

import typing

from ml_switcheroo.core.compiler.frontends.rdna.analysis import RdnaAnalyzer
from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaImmediate as RdnaImm
from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaInstruction as RdnaInst
from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaVGPR as VGPR
from ml_switcheroo.core.compiler.frontends.nvidia_sass.analysis import NvidiaSassAnalyzer
from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassImmediate as NvidiaSassImm
from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassInstruction as NvidiaSassInst
from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassRegister as NvidiaSassReg


def test_nvidia_sass_analyzer_edge_cases() -> None:
  """Verifies the behavior of NVIDIA_SASS analyzer edge cases."""
  assert NvidiaSassAnalyzer.analyze_block("Conv2d", []) == {}
  insts_no_loop = [NvidiaSassInst(opcode="FADD", operands=[NvidiaSassReg(name="R0"), NvidiaSassReg(name="R1")])]
  assert NvidiaSassAnalyzer.analyze_block("Conv2d", insts_no_loop) == {}
  insts_no_imm = [NvidiaSassInst(opcode="ISETP.LT.AND", operands=[NvidiaSassReg(name="R0"), NvidiaSassReg(name="R1")])]
  assert NvidiaSassAnalyzer.analyze_block("Conv2d", insts_no_imm) == {}
  insts_loop = [NvidiaSassInst(opcode="ISETP.LT.AND", operands=[NvidiaSassReg(name="R0"), NvidiaSassImm(value=5)])]
  assert NvidiaSassAnalyzer.analyze_block("Unknown", insts_loop) == {}
  insts_multi = [
    NvidiaSassInst(opcode="ISETP.LT.AND", operands=[NvidiaSassImm(value=3)]),
    NvidiaSassInst(opcode="ISETP.LT.AND", operands=[NvidiaSassImm(value=7)]),
  ]
  meta: dict[str, typing.Any] = NvidiaSassAnalyzer.analyze_block("Conv2d", insts_multi)
  assert meta["kernel_size"] == 7
  assert meta["arg_2"] == 7


def test_rdna_analyzer_edge_cases() -> None:
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
  meta: dict[str, typing.Any] = RdnaAnalyzer.analyze_block("Linear", insts_multi)
  assert meta["in_features"] == 128
  assert meta["arg_0"] == 128
