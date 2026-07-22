"""Test suite for the Mlir Generator Fusion module."""

from ml_switcheroo.core.mlir.generator import MlirToPythonGenerator
from ml_switcheroo.core.mlir.nodes import AttributeNode, BlockNode, ModuleNode, OperationNode, ValueNode


def gen_code(ops: list[OperationNode]) -> str:
  """Helper to generation code."""
  mod = ModuleNode(body=BlockNode(label="", operations=ops))
  gen = MlirToPythonGenerator()
  return gen.generate(mod).code


def test_fusion_return():
  """Verifies the behavior of fusion return."""
  op_call = OperationNode(name="sw.call", results=[ValueNode("%0")], operands=[ValueNode("%fn")])
  op_return = OperationNode(name="sw.return", operands=[ValueNode("%0")])
  code = gen_code([op_call, op_return])
  assert "return _fn()" in code
  assert "=" not in code


def test_fusion_setattr():
  """Verifies the behavior of fusion setattr."""
  op_create = OperationNode(
    name="sw.op", results=[ValueNode("%res")], operands=[ValueNode("%x")], attributes=[AttributeNode("type", '"layer"')]
  )
  op_set = OperationNode(
    name="sw.setattr", operands=[ValueNode("%self"), ValueNode("%res")], attributes=[AttributeNode("name", '"layer"')]
  )
  code = gen_code([op_create, op_set])
  assert "_self.layer = layer(_x)" in code
  assert "_res" not in code


def test_fusion_no_fuse_if_multicount():
  """Verifies the behavior of fusion no fuse if multicount."""
  op_create = OperationNode(
    name="sw.op", results=[ValueNode("%res")], operands=[ValueNode("%x")], attributes=[AttributeNode("type", '"op"')]
  )
  op_set = OperationNode(
    name="sw.setattr", operands=[ValueNode("%self"), ValueNode("%res")], attributes=[AttributeNode("name", '"attr"')]
  )
  op_ret = OperationNode(name="sw.return", operands=[ValueNode("%res")])
  code = gen_code([op_create, op_set, op_ret])
  assert "_op = op(_x)" in code
  assert "_self.attr = _op" in code
  assert "return _op" in code


def test_atom_inlining_getattr():
  """Verifies the behavior of atom inlining getattr."""
  op_get = OperationNode(
    name="sw.getattr",
    results=[ValueNode("%attr")],
    operands=[ValueNode("%self")],
    attributes=[AttributeNode("name", '"conv"')],
  )
  op_call = OperationNode(name="sw.call", results=[ValueNode("%out")], operands=[ValueNode("%attr"), ValueNode("%x")])
  op_ret = OperationNode(name="sw.return", operands=[ValueNode("%out")])
  code = gen_code([op_get, op_call, op_ret])
  assert "_self.conv(_x)" in code
  assert "_attr =" not in code


def test_atom_inlining_constant():
  """Verifies the behavior of atom inlining constant."""
  op_c = OperationNode(name="sw.constant", results=[ValueNode("%c")], attributes=[AttributeNode("value", "1")])
  op_use = OperationNode(
    name="sw.op", results=[ValueNode("%res")], operands=[ValueNode("%c")], attributes=[AttributeNode("type", '"op"')]
  )
  code = gen_code([op_c, op_use])
  assert "op(1)" in code
  assert "_c =" not in code
