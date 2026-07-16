"""Auto-generated doc."""

import pytest

from ml_switcheroo.core.compiler.frontends.rdna.parser import RdnaParser
from ml_switcheroo.core.compiler.frontends.rdna.nodes import Memory, VGPR


def test_rdna_parser_coverage():
  """Auto-generated doc."""
  # line 111, 122, 145
  parser = RdnaParser(".text")
  parser.parse()

  parser = RdnaParser(".amdgcn_target gfx90a\n.text")
  parser.parse()

  parser = RdnaParser("v_add_f32 v0, v1, v2\n.text")
  parser.parse()

  parser = RdnaParser(".directive param1\n.directive2")
  parser.parse()


def test_rdna_parser_memory_operand():
  """Auto-generated doc."""
  # Test parsing of `method[base + offset]` syntax.
  code = "global_load_dword v0, [v1 + 16], off"
  parser = RdnaParser(code)
  nodes = parser.parse()

  assert len(nodes) == 1
  inst = nodes[0]
  assert inst.opcode == "global_load_dword"
  assert len(inst.operands) == 3

  mem_op = inst.operands[1]
  assert isinstance(mem_op, Memory)
  assert isinstance(mem_op.base, VGPR)
  assert mem_op.base.index == 1
  assert mem_op.offset == 16


def test_rdna_parser_memory_syntax_errors():
  """Test syntax errors in memory operand parsing."""
  with pytest.raises(SyntaxError, match="Expected immediate after \\+/\\-"):
    RdnaParser("global_load_dword v0, [v1 + v2], off").parse()

  with pytest.raises(SyntaxError, match="Expected \\]"):
    RdnaParser("global_load_dword v0, [v1 + 16, off").parse()
