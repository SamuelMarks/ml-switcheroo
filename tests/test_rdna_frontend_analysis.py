"""Test suite for rdna frontend analysis."""

from ml_switcheroo.core.compiler.frontends.rdna.analysis import RdnaAnalyzer
from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaImmediate, RdnaInstruction, c_SGPR


def test_analyze_block_empty() -> None:
  """Verify analyze block empty behavior."""
  assert RdnaAnalyzer.analyze_block("Conv2d", []) == {}


def test_analyze_block_no_limits() -> None:
  """Verify analyze block no limits behavior."""
  inst: RdnaInstruction = RdnaInstruction(opcode="v_add_f32", operands=[])
  assert RdnaAnalyzer.analyze_block("Conv2d", [inst]) == {}


def test_analyze_block_conv2d() -> None:
  """Verify analyze block conv2d behavior."""
  inst1: RdnaInstruction = RdnaInstruction(opcode="s_cmp_lt_i32", operands=[c_SGPR(0), RdnaImmediate(value=3)])
  inst2: RdnaInstruction = RdnaInstruction(opcode="s_cmp_lt_i32", operands=[c_SGPR(1), RdnaImmediate(value=5)])
  meta = RdnaAnalyzer.analyze_block("Conv2d", [inst1, inst2])
  assert meta == {"k": 5, "arg_2": 5}


def test_analyze_block_linear() -> None:
  """Verify analyze block linear behavior."""
  inst1: RdnaInstruction = RdnaInstruction(opcode="s_cmp_lt_i32", operands=[c_SGPR(0), RdnaImmediate(value=10)])
  meta = RdnaAnalyzer.analyze_block("Linear", [inst1])
  assert meta == {"in_features": 10, "arg_0": 10}


def test_analyze_block_other() -> None:
  """Verify analyze block other behavior."""
  inst1: RdnaInstruction = RdnaInstruction(opcode="s_cmp_lt_i32", operands=[c_SGPR(0), RdnaImmediate(value=10)])
  meta = RdnaAnalyzer.analyze_block("Other", [inst1])
  assert meta == {}
