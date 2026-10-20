"""Test suite for analysis coverage."""

from ml_switcheroo.core.compiler.frontends.rdna.analysis import RdnaAnalyzer
from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaImmediate, RdnaInstruction, RdnaSGPR


def test_analyze_block_empty() -> None:
  """Verify analyze block empty behavior."""
  res = RdnaAnalyzer.analyze_block("Conv2d", [])
  assert res == {}


def test_analyze_block_conv2d() -> None:
  """Verify analyze block conv2d behavior."""
  inst1 = RdnaInstruction(opcode="s_cmp_lt_i32", operands=[RdnaSGPR(index=0), RdnaImmediate(value=3)])
  inst2 = RdnaInstruction(opcode="s_cmp_lt_i32", operands=[RdnaSGPR(index=1), RdnaImmediate(value=5)])
  inst3 = RdnaInstruction(opcode="s_add_u32", operands=[RdnaSGPR(index=2), RdnaImmediate(value=1)])
  res = RdnaAnalyzer.analyze_block("Conv2d", [inst1, inst2, inst3])
  assert res == {"k": 5, "arg_2": 5}


def test_analyze_block_linear() -> None:
  """Verify analyze block linear behavior."""
  inst = RdnaInstruction(opcode="s_cmp_lt_i32", operands=[RdnaImmediate(value=128)])
  res = RdnaAnalyzer.analyze_block("Linear", [inst])
  assert res == {"in_features": 128, "arg_0": 128}


def test_analyze_block_other() -> None:
  """Verify analyze block other behavior."""
  inst = RdnaInstruction(opcode="s_cmp_lt_i32", operands=[RdnaImmediate(value=10)])
  res = RdnaAnalyzer.analyze_block("UnknownKind", [inst])
  assert res == {}


def test_analyze_block_no_immediate() -> None:
  """Verify analyze block no immediate behavior."""
  inst = RdnaInstruction(opcode="s_cmp_lt_i32", operands=[RdnaSGPR(index=0), RdnaSGPR(index=1)])
  res = RdnaAnalyzer.analyze_block("Conv2d", [inst])
  assert res == {}
