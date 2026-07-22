"""Test suite for the Mlir Generator module."""

from ml_switcheroo.core.mlir.nodes import (
  ModuleNode,
  OperationNode,
  BlockNode,
  RegionNode,
  ValueNode,
  AttributeNode,
  TriviaNode,
  TypeNode,
)
from ml_switcheroo.core.mlir.generator import MlirToPythonGenerator


def gen_code(node: ModuleNode) -> str:
  """Helper to generation code."""
  gen = MlirToPythonGenerator()
  cst_mod = gen.generate(node)
  return cst_mod.code


def test_module_to_class():
  """Verifies the behavior of module to class."""
  op = OperationNode(name="sw.module", attributes=[AttributeNode("sym_name", '"MyClass"')])
  mod = ModuleNode(body=BlockNode(label="", operations=[op]))
  code = gen_code(mod)
  assert "class MyClass:" in code
  assert "pass" in code


def test_func_to_def_with_args():
  """Verifies the behavior of function to def with arguments."""
  ret_op = OperationNode(name="sw.return", operands=[ValueNode("%x")])
  body_blk = BlockNode(label="^entry", arguments=[(ValueNode("%x"), TypeNode("!sw.unk"))], operations=[ret_op])
  func_op = OperationNode(
    name="sw.func", attributes=[AttributeNode("sym_name", '"forward"')], regions=[RegionNode(blocks=[body_blk])]
  )
  mod = ModuleNode(body=BlockNode(label="", operations=[func_op]))
  code = gen_code(mod)
  assert "def forward(x):" in code
  assert "return x" in code


def test_ops_assignment_and_call():
  """Verifies the behavior of ops assignment and call."""
  op = OperationNode(
    name="sw.op",
    results=[ValueNode("%0")],
    operands=[ValueNode("%a"), ValueNode("%b")],
    attributes=[AttributeNode("type", '"torch.add"')],
  )
  use_op = OperationNode(name="sw.return", operands=[ValueNode("%0")])
  mod = ModuleNode(body=BlockNode("", operations=[op, use_op]))
  code = gen_code(mod)
  assert "return torch.add(_a, _b)" in code


def test_trivia_restoration():
  """Verifies the behavior of trivia restoration."""
  op = OperationNode(name="sw.return", leading_trivia=[TriviaNode("// My Comment", "comment")])
  mod = ModuleNode(body=BlockNode("", operations=[op]))
  code = gen_code(mod)
  assert "# My Comment" in code
  assert "return" in code


def test_constant_generation():
  """Verifies the behavior of constant generation."""
  op = OperationNode(name="sw.constant", results=[ValueNode("%c")], attributes=[AttributeNode("value", "1")])
  use_op = OperationNode(name="sw.return", operands=[ValueNode("%c")])
  mod = ModuleNode(body=BlockNode("", operations=[op, use_op]))
  code = gen_code(mod)
  assert "return 1" in code


def test_getattr_generation():
  """Verifies the behavior of getattr generation."""
  op = OperationNode(
    name="sw.getattr",
    results=[ValueNode("%attr")],
    operands=[ValueNode("%self")],
    attributes=[AttributeNode("name", '"layer"')],
  )
  use_op = OperationNode(name="sw.return", operands=[ValueNode("%attr")])
  mod = ModuleNode(body=BlockNode("", operations=[op, use_op]))
  code = gen_code(mod)
  assert "return _self.layer" in code


def test_sw_call_generation():
  """Verifies the behavior of sw call generation."""
  op = OperationNode(name="sw.call", results=[ValueNode("%res")], operands=[ValueNode("%func"), ValueNode("%arg")])
  use_op = OperationNode(name="sw.return", operands=[ValueNode("%res")])
  mod = ModuleNode(body=BlockNode("", operations=[op, use_op]))
  code = gen_code(mod)
  assert "return _func(_arg)" in code
