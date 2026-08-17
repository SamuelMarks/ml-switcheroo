"""Module docstring."""

from ml_switcheroo.core.mlir.cst import (
  TypeNode,
  ValueNode,
  AttributeNode,
  BlockNode,
  RegionNode,
  OperationNode,
  StableHloConstantOp,
  AttributeAliasDefNode,
  ModuleNode,
)
from ml_switcheroo.core.cst.base import Trivia


def test_mlir_cst_coverage():
  """Docstring."""
  t = TypeNode(leading_trivia=[Trivia(" ")], body="i32", trailing_trivia=[Trivia(" ")])
  assert t.to_text() == " i32 "

  v = ValueNode(leading_trivia=[Trivia(" ")], name="%0", type_node=t, colon_trivia=[Trivia(" ")])
  assert v.to_text() == " %0 : i32 "

  v2 = ValueNode(name="%1")
  assert v2.to_text() == "%1"

  a = AttributeNode(leading_trivia=[Trivia(" ")], name="value", value="1", type_annotation="i32")
  assert a.to_text() == " value = 1 : i32"

  a2 = AttributeNode(name="array", value=["1", "2"])
  assert a2.to_text() == "array = [1, 2]"

  a3 = AttributeNode(name="unit", value="")
  assert a3.to_text() == "unit = "

  b = BlockNode(leading_trivia=[Trivia(" ")], label="^bb0", arguments=[(v, t)])
  assert "^bb0" in b.to_text()

  # Missing operation branch
  op_missing = OperationNode(name="missing")
  b3 = BlockNode(label="^bb1", operations=[op_missing])
  b3.to_text()

  b2 = BlockNode(label="")
  assert b2.to_text() == ""

  r = RegionNode(leading_trivia=[Trivia(" ")], blocks=[b2])
  assert r.to_text() == " {}"

  op = OperationNode(
    leading_trivia=[Trivia(" ")],
    name="foo.bar",
    results=[v2],
    operands=[v2],
    attributes=[a3],
    regions=[r],
    result_types=[t],
    op_tail_str=" tail",
    op_tail_trivia=[Trivia(" ")],
    name_trivia=" ",
  )
  text = op.to_text()
  assert "foo.bar" in text

  # check branch in op to_text
  op2 = OperationNode(name="foo", name_trivia=None)
  assert "foo" in op2.to_text()

  # op3 for operands without parens
  op3 = OperationNode(name="foo", operands=[v2, v], has_parens=False, name_trivia="")
  op3.to_text()

  # op4 for operands with parens without name trivia
  op4 = OperationNode(name="foo", operands=[v2, v], has_parens=True, name_trivia="")
  op4.to_text()

  # op5 for op tail str logic
  op5 = OperationNode(name="foo", result_types=[t, t], op_tail_str="", name_trivia="")
  op5.to_text()

  op6 = OperationNode(name="foo", result_types=[t, t], op_tail_str="-> ", name_trivia="")
  op6.to_text()

  op7 = OperationNode(name="foo", result_types=[t, t], op_tail_str=" : ", name_trivia="")
  op7.to_text()

  op8 = OperationNode(name="foo", operands=[v], name_trivia="", has_parens=False)
  op8.to_text()

  op9 = OperationNode(name="foo", operands=[v], name_trivia="", has_parens=True)
  op9.to_text()

  op10 = OperationNode(name="foo", attributes=[a], name_trivia="")
  op10.to_text()

  op11 = OperationNode(name="foo", result_types=[t], op_tail_str="", op_tail_trivia=[], name_trivia="")
  op11.to_text()

  # Test operation with missing region trailing trivia
  op12 = OperationNode(name="foo", regions=[RegionNode(blocks=[b2], leading_trivia=[])], trailing_trivia=[Trivia("t")])
  op12.to_text()

  # Hit branch 170-171, 178
  v_no_trivia = ValueNode(name="%2", leading_trivia=[])
  op13 = OperationNode(name="foo", operands=[v_no_trivia, v_no_trivia], has_parens=False, name_trivia="")
  op13.to_text()

  op14 = OperationNode(name="foo", operands=[v_no_trivia, v_no_trivia], has_parens=True, name_trivia="")
  op14.to_text()

  op15 = OperationNode(name="foo", operands=[v2, v2], has_parens=True, name_trivia="")
  op15.to_text()

  op16 = OperationNode(name="foo", attributes=[a], name_trivia="", has_parens=False)
  op16.to_text()

  op17 = OperationNode(name="foo", result_types=[t], op_tail_str="", op_tail_trivia=[], name_trivia="")
  op17.to_text()

  # More branches for lines 170-171, 173, 189, 200, 253
  # 170-171, 173
  v_empty = ValueNode(name="%3", leading_trivia=[])
  op18 = OperationNode(name="foo", operands=[v_empty], has_parens=False, name_trivia=[])
  op18.to_text()

  op19 = OperationNode(name="foo", operands=[v_empty], has_parens=True, name_trivia=[])
  op19.to_text()

  # 189 (attributes without name trivia)
  op20 = OperationNode(name="foo", attributes=[a], name_trivia=[])
  op20.to_text()

  # 200 (result types without op_tail_trivia and name_trivia)
  op21 = OperationNode(name="foo", result_types=[t], op_tail_trivia=[], name_trivia=[], op_tail_str="")
  op21.to_text()

  # 253 (attributes without name trivia in stablehlo)
  sop6 = StableHloConstantOp(
    name="stablehlo.constant",
    results=[v2],
    attributes=[a],
    result_types=[t],
    name_trivia=[],
    leading_trivia=[],
    trailing_trivia=[],
  )
  sop6.to_text()

  # stablehlo constant branch
  sop = StableHloConstantOp(
    name="stablehlo.constant",
    results=[v2],
    attributes=[AttributeNode(name="value", value="1")],
    result_types=[t],
    name_trivia=" ",
    leading_trivia=[Trivia(" ")],
    trailing_trivia=[Trivia(" ")],
  )
  sop.to_text()

  # stablehlo constant branch 2
  sop2 = StableHloConstantOp(
    name="stablehlo.constant",
    results=[v2],
    attributes=[AttributeNode(name="value", value="1")],
    result_types=[t, t],
    name_trivia="",
    leading_trivia=[],
    trailing_trivia=[],
  )
  sop2.to_text()

  # stablehlo constant branch 3
  sop3 = StableHloConstantOp(
    name="stablehlo.constant",
    results=[v2],
    attributes=[AttributeNode(name="value", value="1")],
    result_types=[],
    name_trivia="",
    leading_trivia=[],
    trailing_trivia=[],
  )
  sop3.to_text()

  # stablehlo constant branch 4
  sop4 = StableHloConstantOp(
    name="stablehlo.constant",
    results=[],
    attributes=[],
    result_types=[],
    name_trivia="",
    leading_trivia=[],
    trailing_trivia=[],
  )
  sop4.to_text()

  # Need to cover the other stablehlo missing lines in to_text
  sop5 = StableHloConstantOp(
    name="stablehlo.constant",
    results=[v2],
    attributes=[],
    result_types=[t, t],
    name_trivia="",
    leading_trivia=[],
    trailing_trivia=[],
  )
  sop5.to_text()

  # alias
  alias = AttributeAliasDefNode(leading_trivia=[Trivia(" ")], name="#map", value_str="affine_map<(d0) -> (d0)>")
  assert alias.to_text() == " #map = affine_map<(d0) -> (d0)>"

  alias2 = AttributeAliasDefNode(name="#map", value_node=TypeNode(body="i32"))
  assert alias2.to_text() == "#map = i32"

  m = ModuleNode(leading_trivia=[Trivia(" ")], body=b2, aliases=[alias])
  assert " #map = affine_map<(d0) -> (d0)>" in m.to_text()
