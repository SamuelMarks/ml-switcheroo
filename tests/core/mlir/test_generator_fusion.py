"""
Tests for Statement Fusion Logic in MLIR Generator.

Verifies that operations consumed immediately by `setattr` or `return` are
inlined regardless of the `inline_expressions` flag, creating standard
Python/PyTorch idiom code (e.g. `self.x = y` instead of `_0=y; self.x=_0`).
"""

from ml_switcheroo.core.mlir.generator import MlirToPythonGenerator
from ml_switcheroo.core.mlir.nodes import (
  AttributeNode,
  BlockNode,
  ModuleNode,
  OperationNode,
  ValueNode,
)


def gen_code(ops: list[OperationNode]) -> str:
  """Generates Python code from a flat list of ops (default inline=False)."""
  mod = ModuleNode(body=BlockNode(label="", operations=ops))
  gen = MlirToPythonGenerator(inline_expressions=False)
  return gen.generate(mod).code


def test_fusion_return():
  """
  Scenario: %0 = call(); return %0
  Mode: inline=False
  Expect: return call() (Fused), NOT _0=call(); return _0
  """
  op_call = OperationNode(name="sw.call", results=[ValueNode("%0")], operands=[ValueNode("%fn")])
  op_return = OperationNode(name="sw.return", operands=[ValueNode("%0")])

  code = gen_code([op_call, op_return])

  assert "return _fn()" in code
  assert "=" not in code


def test_fusion_setattr():
  """
  Scenario: %0 = op(); setattr(%self, "attr", %0)
  Mode: inline=False
  Expect: self.attr = op() (Fused)
  """
  op_create = OperationNode(
    name="sw.op", results=[ValueNode("%res")], operands=[ValueNode("%x")], attributes=[AttributeNode("type", '"layer"')]
  )

  op_set = OperationNode(
    name="sw.setattr", operands=[ValueNode("%self"), ValueNode("%res")], attributes=[AttributeNode("name", '"layer"')]
  )

  code = gen_code([op_create, op_set])

  # Check cleaner syntax
  assert "_self.layer = layer(_x)" in code
  assert "_res" not in code  # Intermediate should be gone


def test_fusion_no_fuse_if_multicount():
  """
  Scenario: %0 = op(); setattr(..., %0); return %0
  Usage Count: 2
  Mode: inline=False
  Expect: NO Fusion. _op = op(); self.a = _op; return _op

  Note: Variable name '_op' is derived from the type attribute "op".
  """
  op_create = OperationNode(
    name="sw.op", results=[ValueNode("%res")], operands=[ValueNode("%x")], attributes=[AttributeNode("type", '"op"')]
  )
  op_set = OperationNode(
    name="sw.setattr", operands=[ValueNode("%self"), ValueNode("%res")], attributes=[AttributeNode("name", '"attr"')]
  )
  op_ret = OperationNode(name="sw.return", operands=[ValueNode("%res")])

  code = gen_code([op_create, op_set, op_ret])

  # Must assign intermediate because it is used twice
  # Variable name defaults to Semantic Hint derived from type ("op") -> _op
  assert "_op = op(_x)" in code
  assert "_self.attr = _op" in code
  assert "return _op" in code


def test_atom_inlining_getattr():
  """
  Scenario: %0 = getattr(%self, "conv"); %1 = call(%0, %x)
  Mode: inline=False
  Expect: self.conv(x) (getattr is always atomic/inlined)
  """
  op_get = OperationNode(
    name="sw.getattr",
    results=[ValueNode("%attr")],
    operands=[ValueNode("%self")],
    attributes=[AttributeNode("name", '"conv"')],
  )
  op_call = OperationNode(name="sw.call", results=[ValueNode("%out")], operands=[ValueNode("%attr"), ValueNode("%x")])
  op_ret = OperationNode(name="sw.return", operands=[ValueNode("%out")])

  code = gen_code([op_get, op_call, op_ret])

  # _self.conv should be inlined into the call
  assert "_self.conv(_x)" in code
  # No assignment for getattr intermediate
  assert "_attr =" not in code


def test_atom_inlining_constant():
  """
  Scenario: %c = constant 1; op(%c)
  Mode: inline=False
  Expect: op(1)
  """
  op_c = OperationNode(name="sw.constant", results=[ValueNode("%c")], attributes=[AttributeNode("value", "1")])
  op_use = OperationNode(
    name="sw.op", results=[ValueNode("%res")], operands=[ValueNode("%c")], attributes=[AttributeNode("type", '"op"')]
  )

  code = gen_code([op_c, op_use])

  assert "op(1)" in code
  assert "_c =" not in code
