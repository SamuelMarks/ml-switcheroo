"""Tests for RDNA analysis coverage."""

import typing
from ml_switcheroo.core.compiler.frontends.rdna.analysis import RdnaAnalyzer
from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaInstruction, RdnaImmediate, RdnaSGPR


def test_analyze_block_empty() -> None:
  """Test analyzing an empty block."""
  res: dict[str, typing.Any] = RdnaAnalyzer.analyze_block("Conv2d", [])
  assert res == {}


def test_analyze_block_conv2d() -> None:
  """Test analyzing a Conv2d block."""
  inst1 = RdnaInstruction(opcode="s_cmp_lt_i32", operands=[RdnaSGPR(index=0), RdnaImmediate(value=3)])
  inst2 = RdnaInstruction(opcode="s_cmp_lt_i32", operands=[RdnaSGPR(index=1), RdnaImmediate(value=5)])
  inst3 = RdnaInstruction(opcode="s_add_u32", operands=[RdnaSGPR(index=2), RdnaImmediate(value=1)])
  res: dict[str, typing.Any] = RdnaAnalyzer.analyze_block("Conv2d", [inst1, inst2, inst3])
  assert res == {"k": 5, "arg_2": 5}


def test_analyze_block_linear() -> None:
  """Test analyzing a Linear block."""
  inst = RdnaInstruction(opcode="s_cmp_lt_i32", operands=[RdnaImmediate(value=128)])
  res: dict[str, typing.Any] = RdnaAnalyzer.analyze_block("Linear", [inst])
  assert res == {"in_features": 128, "arg_0": 128}


def test_analyze_block_other() -> None:
  """Test analyzing an unknown kind block."""
  inst = RdnaInstruction(opcode="s_cmp_lt_i32", operands=[RdnaImmediate(value=10)])
  res: dict[str, typing.Any] = RdnaAnalyzer.analyze_block("UnknownKind", [inst])
  assert res == {}


def test_analyze_block_no_immediate() -> None:
  """Test analyzing a block with no immediate operands."""
  inst = RdnaInstruction(opcode="s_cmp_lt_i32", operands=[RdnaSGPR(index=0), RdnaSGPR(index=1)])
  res: dict[str, typing.Any] = RdnaAnalyzer.analyze_block("Conv2d", [inst])
  assert res == {}
