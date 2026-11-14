"""Unit tests for RDNA instruction block loop/parameter analysis."""

from __future__ import annotations

from ml_switcheroo.core.compiler.frontends.rdna.analysis import RdnaAnalyzer
from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaImmediate, RdnaInstruction, RdnaSGPR, c_SGPR


def test_analyze_block_empty() -> None:
  """Verify RdnaAnalyzer returns empty metadata for an empty instruction list."""
  assert RdnaAnalyzer.analyze_block("Conv2d", []) == {}


def test_analyze_block_no_limits() -> None:
  """Verify RdnaAnalyzer returns empty metadata when no loop-comparison opcodes exist."""
  inst = RdnaInstruction(opcode="v_add_f32", operands=[])
  assert RdnaAnalyzer.analyze_block("Conv2d", [inst]) == {}


def test_analyze_block_conv2d() -> None:
  """Verify RdnaAnalyzer extracts kernel size k from Conv2d s_cmp_lt_i32 loop bounds."""
  inst1 = RdnaInstruction(opcode="s_cmp_lt_i32", operands=[c_SGPR(0), RdnaImmediate(value=3)])
  inst2 = RdnaInstruction(opcode="s_cmp_lt_i32", operands=[RdnaSGPR(index=1), RdnaImmediate(value=5)])
  inst3 = RdnaInstruction(opcode="s_add_u32", operands=[RdnaSGPR(index=2), RdnaImmediate(value=1)])
  assert RdnaAnalyzer.analyze_block("Conv2d", [inst1, inst2, inst3]) == {"k": 5, "arg_2": 5}


def test_analyze_block_linear() -> None:
  """Verify RdnaAnalyzer extracts in_features from Linear s_cmp_lt_i32 loop bounds."""
  inst = RdnaInstruction(opcode="s_cmp_lt_i32", operands=[c_SGPR(0), RdnaImmediate(value=128)])
  assert RdnaAnalyzer.analyze_block("Linear", [inst]) == {"in_features": 128, "arg_0": 128}


def test_analyze_block_other() -> None:
  """Verify RdnaAnalyzer returns empty metadata for unrecognized operation kinds."""
  inst = RdnaInstruction(opcode="s_cmp_lt_i32", operands=[c_SGPR(0), RdnaImmediate(value=10)])
  assert RdnaAnalyzer.analyze_block("UnknownKind", [inst]) == {}


def test_analyze_block_no_immediate() -> None:
  """Verify RdnaAnalyzer returns empty metadata when s_cmp_lt_i32 lacks an immediate operand."""
  inst = RdnaInstruction(opcode="s_cmp_lt_i32", operands=[RdnaSGPR(index=0), RdnaSGPR(index=1)])
  assert RdnaAnalyzer.analyze_block("Conv2d", [inst]) == {}
