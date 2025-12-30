"""
Tests for MLIR -> Python Generator.

Verifies:
1.  Structure: sw.module -> class, sw.func -> def.
2.  Logic: Assignments, Returns, Calls.
3.  Naming: SSA IDs mapped to plausible Python variables.
4.  Trivia: Comments restored.
"""

import pytest
import libcst as cst
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
  gen = MlirToPythonGenerator()
  cst_mod = gen.generate(node)
  return cst_mod.code


def test_module_to_class():
  """
  Input: sw.module {sym_name = "MyClass"} {}
  Expect: class MyClass: pass
  """
  op = OperationNode(name="sw.module", attributes=[AttributeNode("sym_name", '"MyClass"')])
  # Wrap in module structure
  mod = ModuleNode(body=BlockNode(label="", operations=[op]))

  code = gen_code(mod)
  assert "class MyClass:" in code
  assert "pass" in code


def test_func_to_def_with_args():
  """
  Input: sw.func {sym_name="forward"} ^entry(%x: !sw.unk): ...
  Expect: def forward(x): ...
  """
  # 1. Create Func Body Block
  # Need at least one op or pass logic handles it
  ret_op = OperationNode(name="sw.return", operands=[ValueNode("%x")])

  body_blk = BlockNode(label="^entry", arguments=[(ValueNode("%x"), TypeNode("!sw.unk"))], operations=[ret_op])

  # 2. Create Func Op
  func_op = OperationNode(
    name="sw.func", attributes=[AttributeNode("sym_name", '"forward"')], regions=[RegionNode(blocks=[body_blk])]
  )

  mod = ModuleNode(body=BlockNode(label="", operations=[func_op]))
  code = gen_code(mod)

  # Check def signature
  assert "def forward(x):" in code
  # Check naming context mapping (%x -> x)
  assert "return x" in code


def test_ops_assignment_and_call():
  """
  Input:
      %0 = sw.op {type="torch.add"} (%a, %b)
  Expect:
      v0 = torch.add(a, b)
  """
  # Requires context seeding for %a, %b or we assume generator handles lookup safely
  # If they aren't registered, lookup returns modified string.

  op = OperationNode(
    name="sw.op",
    results=[ValueNode("%0")],
    operands=[ValueNode("%a"), ValueNode("%b")],
    attributes=[AttributeNode("type", '"torch.add"')],
  )

  mod = ModuleNode(body=BlockNode("", operations=[op]))
  code = gen_code(mod)

  # Generator defaults unregistered names: %a -> _a (if reserved) or just a
  # NamingContext.lookup("%a") -> "_a" (fallback replaces % with _)
  assert "_0 = torch.add(_a, _b)" in code


def test_trivia_restoration():
  """
  Input: // My Comment attached to op
  Expect: # My Comment
  """
  op = OperationNode(name="sw.return", leading_trivia=[TriviaNode("// My Comment", "comment")])
  mod = ModuleNode(body=BlockNode("", operations=[op]))
  code = gen_code(mod)

  assert "# My Comment" in code
  assert "return" in code


def test_constant_generation():
  """
  Input: %c = sw.constant {value = 1}
  Expect: v0 = 1
  """
  op = OperationNode(name="sw.constant", results=[ValueNode("%c")], attributes=[AttributeNode("value", "1")])
  mod = ModuleNode(body=BlockNode("", operations=[op]))
  code = gen_code(mod)

  assert "c = 1" in code or "_c = 1" in code


def test_getattr_generation():
  """
  Input:
      %attr = sw.getattr %self {name = "layer"}
  Expect:
      layer = self.layer
  """
  op = OperationNode(
    name="sw.getattr",
    results=[ValueNode("%attr")],
    operands=[ValueNode("%self")],
    attributes=[AttributeNode("name", '"layer"')],
  )

  mod = ModuleNode(body=BlockNode("", operations=[op]))
  code = gen_code(mod)

  # Heuristic uses hint="layer" for result name
  assert "layer = _self.layer" in code


def test_sw_call_generation():
  """
  Input: %res = sw.call %func (%arg)
  Expect: res = _func(_arg)
  """
  op = OperationNode(name="sw.call", results=[ValueNode("%res")], operands=[ValueNode("%func"), ValueNode("%arg")])
  mod = ModuleNode(body=BlockNode("", operations=[op]))
  code = gen_code(mod)

  assert "res = _func(_arg)" in code
