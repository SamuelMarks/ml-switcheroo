"""Test suite for the Mlir Generator module."""

from ml_switcheroo.core.cst.base import Trivia
from ml_switcheroo.core.mlir.cst import (
  ModuleNode,
  OperationNode,
  BlockNode,
  RegionNode,
  ValueNode,
  AttributeNode,
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
  op = OperationNode(name="sw.module", attributes=[AttributeNode(name="sym_name", value='"MyClass"')])
  mod = ModuleNode(body=BlockNode(label="", operations=[op]))
  code = gen_code(mod)
  assert "class MyClass:" in code
  assert "pass" in code


def test_func_to_def_with_args():
  """Verifies the behavior of function to def with arguments."""
  ret_op = OperationNode(name="sw.return", operands=[ValueNode(name="%x")])
  body_blk = BlockNode(label="^entry", arguments=[(ValueNode(name="%x"), TypeNode(body="!sw.unk"))], operations=[ret_op])
  func_op = OperationNode(
    name="sw.func",
    attributes=[AttributeNode(name="sym_name", value='"forward"')],
    regions=[RegionNode(blocks=[body_blk])],
  )
  mod = ModuleNode(body=BlockNode(label="", operations=[func_op]))
  code = gen_code(mod)
  assert "def forward(x):" in code
  assert "return x" in code


def test_ops_assignment_and_call():
  """Verifies the behavior of ops assignment and call."""
  op = OperationNode(
    name="sw.op",
    results=[ValueNode(name="%0")],
    operands=[ValueNode(name="%a"), ValueNode(name="%b")],
    attributes=[AttributeNode(name="type", value='"torch.add"')],
  )
  use_op = OperationNode(name="sw.return", operands=[ValueNode(name="%0")])
  mod = ModuleNode(body=BlockNode("", operations=[op, use_op]))
  code = gen_code(mod)
  assert "return torch.add(_a, _b)" in code


def test_trivia_restoration():
  """Verifies the behavior of trivia restoration."""
  op = OperationNode(name="sw.return", leading_trivia=[Trivia("// My Comment")])
  mod = ModuleNode(body=BlockNode("", operations=[op]))
  code = gen_code(mod)
  assert "# My Comment" in code
  assert "return" in code


def test_constant_generation():
  """Verifies the behavior of constant generation."""
  op = OperationNode(
    name="sw.constant", results=[ValueNode(name="%c")], attributes=[AttributeNode(name="value", value="1")]
  )
  use_op = OperationNode(name="sw.return", operands=[ValueNode(name="%c")])
  mod = ModuleNode(body=BlockNode("", operations=[op, use_op]))
  code = gen_code(mod)
  assert "return 1" in code


def test_getattr_generation():
  """Verifies the behavior of getattr generation."""
  op = OperationNode(
    name="sw.getattr",
    results=[ValueNode(name="%attr")],
    operands=[ValueNode(name="%self")],
    attributes=[AttributeNode(name="name", value='"layer"')],
  )
  use_op = OperationNode(name="sw.return", operands=[ValueNode(name="%attr")])
  mod = ModuleNode(body=BlockNode("", operations=[op, use_op]))
  code = gen_code(mod)
  assert "return _self.layer" in code


def test_sw_call_generation():
  """Verifies the behavior of sw call generation."""
  op = OperationNode(
    name="sw.call", results=[ValueNode(name="%res")], operands=[ValueNode(name="%func"), ValueNode(name="%arg")]
  )
  use_op = OperationNode(name="sw.return", operands=[ValueNode(name="%res")])
  mod = ModuleNode(body=BlockNode("", operations=[op, use_op]))
  code = gen_code(mod)
  assert "return _func(_arg)" in code
