"""Test suite for the Mlir Generator Hardening module."""

from ml_switcheroo.core.mlir.cst import AttributeNode, BlockNode, ModuleNode, OperationNode, ValueNode
from ml_switcheroo.core.mlir.generator import MlirToPythonGenerator


def gen_code(op: OperationNode) -> str:
  """Execute `ModuleNode` to support gen code in MLIR generator hardening tests.

  Args:
      op: Input `op` parameter value (`OperationNode`).

  Returns:
      Resulting `str` from gen code.
  """
  mod = ModuleNode(body=BlockNode(label="", operations=[op]))
  gen = MlirToPythonGenerator()
  return gen.generate(mod).code


def test_sw_op_attribute_hardening() -> None:
  """Verify `OperationNode` emits `torch.nn.Conv2d(_arg)` for sw operation attribute hardening."""
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
  """Verify `MlirToPythonGenerator` emits `self.conv(_x)` for sw call method chain hardening."""
  gen = MlirToPythonGenerator()
  gen.ctx._map["%self_conv"] = "self.conv"
  op = OperationNode(
    name="sw.call", results=[ValueNode(name="%res")], operands=[ValueNode(name="%self_conv"), ValueNode(name="%x")]
  )
  mod = ModuleNode(body=BlockNode(label="", operations=[op]))
  code: str = gen.generate(mod).code
  assert "self.conv(_x)" in code
  assert "=" not in code


def test_sw_op_void_return() -> None:
  """Verify `OperationNode` emits `print(_arg)` for sw operation void return."""
  op = OperationNode(
    name="sw.op", results=[], attributes=[AttributeNode(name="type", value='"print"')], operands=[ValueNode(name="%arg")]
  )
  code: str = gen_code(op)
  assert "print(_arg)" in code
  assert "=" not in code


def test_naming_context_reserved_words() -> None:
  """Verify `MlirToPythonGenerator` handles naming context reserved words as expected."""
  gen = MlirToPythonGenerator()
  name: str = gen.ctx.register("%0", hint="for")
  assert name == "_for" or name == "v_for"


def test_naming_context_global_symbol() -> None:
  """Verify `MlirToPythonGenerator` handles naming context global symbol as expected."""
  gen = MlirToPythonGenerator()
  name: str = gen.ctx.lookup("@my_global_func")
  assert name == "my_global_func"
