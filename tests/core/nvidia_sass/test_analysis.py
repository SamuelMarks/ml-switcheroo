"""Test suite for the Analysis module."""

import typing

from ml_switcheroo.core.compiler.frontends.nvidia_sass.analysis import NvidiaSassAnalyzer
from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import (
  NvidiaSassImmediate,
  NvidiaSassInstruction,
  NvidiaSassLabel,
  NvidiaSassMemory,
  NvidiaSassPredicate,
  NvidiaSassRegister,
)


def make_inst(opcode: str, *operands: typing.Any) -> NvidiaSassInstruction:
  """Helper to make inst."""
  return NvidiaSassInstruction(opcode=opcode, operands=list(operands))


def test_analyze_conv2d_kernel_size() -> None:
  """Analyzes conv2d kernel size."""
  r3 = NvidiaSassRegister(name="R3")
  pt = NvidiaSassRegister(name="PT")
  p0 = NvidiaSassPredicate(name="P0")
  insts = [
    make_inst("MOV", NvidiaSassRegister(name="R1"), NvidiaSassRegister(name="RZ")),
    make_inst("ISETP.LT.AND", p0, pt, r3, NvidiaSassImmediate(value=3), pt),
    make_inst("BRA", NvidiaSassLabel(name="L_LOOP")),  # type: ignore
  ]
  meta: dict[str, typing.Any] = NvidiaSassAnalyzer.analyze_block("Conv2d", insts)
  assert "kernel_size" in meta
  assert meta["kernel_size"] == 3
  assert meta["arg_2"] == 3


def test_analyze_linear_in_features() -> None:
  """Analyzes linear in features."""
  r8 = NvidiaSassRegister(name="R8")
  pt = NvidiaSassRegister(name="PT")
  p0 = NvidiaSassPredicate(name="P0")
  insts = [
    make_inst("LDG.E.F32", NvidiaSassRegister(name="R9"), NvidiaSassMemory(base="addr")),  # type: ignore
    make_inst("ISETP.LT.AND", p0, pt, r8, NvidiaSassImmediate(value=128), pt),
  ]
  meta: dict[str, typing.Any] = NvidiaSassAnalyzer.analyze_block("Linear", insts)
  assert "in_features" in meta
  assert meta["in_features"] == 128
  assert meta["arg_0"] == 128


def test_analyze_no_loop_found() -> None:
  """Analyzes no loop found."""
  insts = [make_inst("FADD", NvidiaSassRegister(name="R0"), NvidiaSassRegister(name="R1"), NvidiaSassRegister(name="R2"))]
  meta: dict[str, typing.Any] = NvidiaSassAnalyzer.analyze_block("Linear", insts)
  assert meta == {}


def test_nvidia_sass_analysis_other_kinds() -> None:
  """Docstring."""
  from ml_switcheroo.core.compiler.frontends.nvidia_sass.analysis import NvidiaSassAnalyzer
  from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassImmediate, NvidiaSassInstruction

  inst = NvidiaSassInstruction(opcode="ISETP.LT.AND", operands=[NvidiaSassImmediate(value=5)])

  meta: dict[str, typing.Any] = NvidiaSassAnalyzer.analyze_block("Conv3d", [inst])
  assert meta["kernel_size"] == 5

  meta = NvidiaSassAnalyzer.analyze_block("AvgPool2d", [inst])
  assert meta["kernel_size"] == 5

  NvidiaSassAnalyzer.analyze_block("BatchNorm2d", [inst])
  NvidiaSassAnalyzer.analyze_block("Conv1d", [inst])
  NvidiaSassAnalyzer.analyze_block("BatchNorm1d", [inst])
  NvidiaSassAnalyzer.analyze_block("Softmax", [inst])
  NvidiaSassAnalyzer.analyze_block("BMM", [inst])
  NvidiaSassAnalyzer.analyze_block("Sum", [inst])
  NvidiaSassAnalyzer.analyze_block("BCEWithLogitsLoss", [inst])
  NvidiaSassAnalyzer.analyze_block("Dropout2d", [inst])
  NvidiaSassAnalyzer.analyze_block("AvgPool1d", [inst])
  NvidiaSassAnalyzer.analyze_block("MultiheadAttention", [inst])
  NvidiaSassAnalyzer.analyze_block("RNN", [inst])

  meta = NvidiaSassAnalyzer.analyze_block("MSELoss", [inst])
  assert meta["elements"] == 5

  NvidiaSassAnalyzer.analyze_block("Sigmoid", [inst])
  NvidiaSassAnalyzer.analyze_block("Dropout", [inst])

  meta = NvidiaSassAnalyzer.analyze_block("Mean", [inst])
  assert meta["elements"] == 5
