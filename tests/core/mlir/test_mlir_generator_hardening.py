"""Test suite for the Mlir Generator Hardening module."""

import typing

from ml_switcheroo.core.mlir.cst import AttributeNode, BlockNode, ModuleNode, OperationNode, ValueNode
from ml_switcheroo.core.mlir.generator import MlirToPythonGenerator


def gen_code(op: OperationNode) -> str:
  """Helper to generation code."""
  mod = ModuleNode(body=BlockNode(label="", operations=[op]))  # type: ignore
  gen = MlirToPythonGenerator()
  return typing.cast(str, gen.generate(mod).code)


def test_sw_op_attribute_hardening() -> None:
  """Verifies the behavior of sw op attribute hardening."""
  op = OperationNode(
    name="sw.op",
    results=[ValueNode(name="%res")],
    attributes=[AttributeNode(name="type", value='"torch.nn.Conv2d"')],
    operands=[ValueNode(name="%arg")],
  )
  code: str = gen_code(op)
  assert "torch.nn.Conv2d(_arg)" in code
  assert "=" not in code


def test_sw_call_method_chain_hardening() -> None:
  """Verifies the behavior of sw call method chain hardening."""
  gen = MlirToPythonGenerator()
  gen.ctx._map["%self_conv"] = "self.conv"
  op = OperationNode(
    name="sw.call", results=[ValueNode(name="%res")], operands=[ValueNode(name="%self_conv"), ValueNode(name="%x")]
  )
  mod = ModuleNode(body=BlockNode(label="", operations=[op]))  # type: ignore
  code: str = typing.cast(str, gen.generate(mod).code)
  assert "self.conv(_x)" in code
  assert "=" not in code


def test_sw_op_void_return() -> None:
  """Verifies the behavior of sw op void return."""
  op = OperationNode(
    name="sw.op", results=[], attributes=[AttributeNode(name="type", value='"print"')], operands=[ValueNode(name="%arg")]
  )
  code: str = gen_code(op)
  assert "print(_arg)" in code
  assert "=" not in code


def test_naming_context_reserved_words() -> None:
  """Verifies the behavior of naming context reserved words."""
  gen = MlirToPythonGenerator()
  name: str = gen.ctx.register("%0", hint="for")
  assert name == "_for" or name == "v_for"


def test_naming_context_global_symbol() -> None:
  """Verifies the behavior of naming context global symbol."""
  gen = MlirToPythonGenerator()
  name: str = gen.ctx.lookup("@my_global_func")  # type: ignore
  assert name == "my_global_func"
