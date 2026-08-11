"""Tests."""

import pytest
from ml_switcheroo.core.compiler.backends.rdna.synthesizer import RegisterAllocator, RdnaSynthesizer
from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaImmediate, RdnaLabel, RdnaOperand, RdnaInstruction


def test_register_allocator_overflow_vgpr():
  """Test function."""
  allocator = RegisterAllocator()
  allocator._next_vgpr = 256
  with pytest.raises(ValueError, match="RdnaVGPR overflow"):
    allocator.get_vector_register("test")


def test_register_allocator_overflow_sgpr():
  """Test function."""
  allocator = RegisterAllocator()
  allocator._next_sgpr = 106
  with pytest.raises(ValueError, match="RdnaSGPR overflow"):
    allocator.get_scalar_register("test")


def test_convert_operand_to_py_immediate_float():
  """Test function."""
  synth = RdnaSynthesizer(None)
  imm = RdnaImmediate(value=3.14)
  res = synth._convert_operand_to_py(imm)
  assert getattr(res, "value", None) == "3.14"


def test_convert_operand_to_py_brackets():
  """Test function."""
  synth = RdnaSynthesizer(None)

  class DummyOp(RdnaOperand):
    def __str__(self):
      """Test function."""
      return "v[1:2]"

  res = synth._convert_operand_to_py(DummyOp())
  assert getattr(res, "value", None) == "v_1_2"


def test_convert_operand_to_py_fallback():
  """Test function."""
  synth = RdnaSynthesizer(None)

  class DummyOp(RdnaOperand):
    def __str__(self):
      """Test function."""
      return "some-weird-str!"

  res = synth._convert_operand_to_py(DummyOp())
  assert res.value == "'some-weird-str!'"


def test_rdna_synthesizer_label_conversion():
  """Test function."""
  synth = RdnaSynthesizer(None)
  label = RdnaLabel(name="my_label")
  mod = synth.to_python([label])
  assert mod is not None


def test_convert_instruction_to_py_no_operands():
  """Test function."""
  synth = RdnaSynthesizer(None)
  inst = RdnaInstruction(opcode="s_endpgm", operands=[])
  res = synth._convert_instruction_to_py(inst)
  assert res is not None
