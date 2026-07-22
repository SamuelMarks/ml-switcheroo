"""Test suite for the Mlir Generator Hardening module."""

from ml_switcheroo.core.mlir.generator import MlirToPythonGenerator
from ml_switcheroo.core.mlir.nodes import AttributeNode, BlockNode, ModuleNode, OperationNode, ValueNode


def gen_code(op: OperationNode) -> str:
  """Helper to generation code."""
  mod = ModuleNode(body=BlockNode(label="", operations=[op]))
  gen = MlirToPythonGenerator()
  return gen.generate(mod).code


def test_sw_op_attribute_hardening():
  """Verifies the behavior of sw op attribute hardening."""
  op = OperationNode(
    name="sw.op",
    results=[ValueNode("%res")],
    attributes=[AttributeNode("type", '"torch.nn.Conv2d"')],
    operands=[ValueNode("%arg")],
  )
  code = gen_code(op)
  assert "torch.nn.Conv2d(_arg)" in code
  assert "=" not in code


def test_sw_call_method_chain_hardening():
  """Verifies the behavior of sw call method chain hardening."""
  gen = MlirToPythonGenerator()
  gen.ctx._map["%self_conv"] = "self.conv"
  op = OperationNode(name="sw.call", results=[ValueNode("%res")], operands=[ValueNode("%self_conv"), ValueNode("%x")])
  mod = ModuleNode(body=BlockNode(label="", operations=[op]))
  code = gen.generate(mod).code
  assert "self.conv(_x)" in code
  assert "=" not in code


def test_sw_op_void_return():
  """Verifies the behavior of sw op void return."""
  op = OperationNode(
    name="sw.op", results=[], attributes=[AttributeNode("type", '"print"')], operands=[ValueNode("%arg")]
  )
  code = gen_code(op)
  assert "print(_arg)" in code
  assert "=" not in code


def test_naming_context_reserved_words():
  """Verifies the behavior of naming context reserved words."""
  gen = MlirToPythonGenerator()
  name = gen.ctx.register("%0", hint="for")
  assert name == "_for" or name == "v_for"


def test_naming_context_global_symbol():
  """Verifies the behavior of naming context global symbol."""
  gen = MlirToPythonGenerator()
  name = gen.ctx.lookup("@my_global_func")
  assert name == "my_global_func"
