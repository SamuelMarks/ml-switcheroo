"""
Tests for Generator Hardening requirements.
Efficiently validates sw.op attribute handling and call structures.
"""

from ml_switcheroo.core.mlir.generator import MlirToPythonGenerator
from ml_switcheroo.core.mlir.nodes import (
  AttributeNode,
  BlockNode,
  ModuleNode,
  OperationNode,
  ValueNode,
)


def gen_code(op: OperationNode) -> str:
  """Helper to generate python code from a single Op node (wrapped in module)."""
  mod = ModuleNode(body=BlockNode(label="", operations=[op]))
  gen = MlirToPythonGenerator()
  return gen.generate(mod).code


def test_sw_op_attribute_hardening():
  """
  Verify `sw.op` uses `type` attribute for complex dotted names.
  Scenario: x = torch.nn.Conv2d(...)
  """
  op = OperationNode(
    name="sw.op",
    results=[ValueNode("%res")],
    attributes=[AttributeNode("type", '"torch.nn.Conv2d"')],
    operands=[ValueNode("%arg")],
  )

  code = gen_code(op)
  assert "conv2d = torch.nn.Conv2d(_arg)" in code


def test_sw_call_method_chain_hardening():
  """
  Verify `sw.call` correctly handles callable stored in variable.
  Scenario:
      self.conv = ...
      x = self.conv(x)
  """
  # 1. Simulate naming context having 'self.conv' registered
  gen = MlirToPythonGenerator()
  gen.ctx._map["%self_conv"] = "self.conv"

  # 2. Define %res = sw.call %self_conv (%x)
  op = OperationNode(
    name="sw.call",
    results=[ValueNode("%res")],
    operands=[ValueNode("%self_conv"), ValueNode("%x")],
  )

  mod = ModuleNode(body=BlockNode(label="", operations=[op]))
  code = gen.generate(mod).code

  # Expect it to resolve 'self.conv' into proper CST attributes
  assert "_res = self.conv(_x)" in code


def test_sw_op_void_return():
  """
  Verify `sw.op` without results generates an Expression Statement.
  Scenario: print("hello")
  """
  op = OperationNode(
    name="sw.op",
    results=[],
    attributes=[AttributeNode("type", '"print"')],
    operands=[ValueNode("%arg")],
  )

  code = gen_code(op)
  # verify strictly no assignment "="
  assert "print(_arg)" in code
  assert "=" not in code


def test_naming_context_reserved_words():
  """
  Verify NamingContext avoids keywords if they appear as SSA hints.
  """
  gen = MlirToPythonGenerator()
  # %0 -> hint="for" -> invalid python var
  name = gen.ctx.register("%0", hint="for")
  assert name == "_for" or name == "v_for"


def test_naming_context_global_symbol():
  """
  Verify global symbol references (@func) are stripped.
  """
  gen = MlirToPythonGenerator()
  name = gen.ctx.lookup("@my_global_func")
  assert name == "my_global_func"
