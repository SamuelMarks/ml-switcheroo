"""Unit tests for MLIR Concrete Syntax Tree (CST) nodes (`ml_switcheroo.core.mlir.cst`)."""

import pytest

from ml_switcheroo.core.cst.base import Trivia
from ml_switcheroo.core.mlir.cst import (
  AttributeAliasDefNode,
  AttributeNode,
  BlockNode,
  MlirNode,
  ModuleNode,
  OperationNode,
  RegionNode,
  StableHloConstantOp,
  TriviaNode,
  TypeAliasDefNode,
  TypeNode,
  ValueNode,
)

_NL = chr(10)


def test_mlir_node_base_and_trivia_aliases() -> None:
  """Verify abstract `MlirNode.to_text()` raises `NotImplementedError` and `TriviaNode` formats text."""

  class UnimplementedMlirNode(MlirNode):
    """Concrete subclass without `to_text` override."""

  with pytest.raises(NotImplementedError):
    UnimplementedMlirNode().to_text()

  assert TriviaNode(_NL).to_text() == _NL
  assert TriviaNode("// comment").to_text() == "// comment"


def test_value_type_attribute_and_alias_nodes() -> None:
  """Verify `TypeNode`, `ValueNode`, `AttributeNode`, `AttributeAliasDefNode`, and `TypeAliasDefNode` formatting."""
  t = TypeNode(leading_trivia=[Trivia(" ")], body="i32", trailing_trivia=[Trivia(" ")])
  assert t.to_text() == " i32 "

  v_typed = ValueNode(leading_trivia=[Trivia(" ")], name="%0", type_node=t, colon_trivia=[Trivia(" ")])
  assert v_typed.to_text() == " %0 : i32 "

  v_indexed = ValueNode(name="%0", use_index=1, trailing_trivia=[Trivia(" ")])
  assert v_indexed.to_text() == "%0#1 "

  attr_typed = AttributeNode(leading_trivia=[Trivia(" ")], name="value", value="1", type_annotation="i32")
  assert attr_typed.to_text() == " value = 1 : i32"

  attr_list = AttributeNode(name="array", value=["1", "2"], trailing_trivia=[Trivia(" ")])
  assert attr_list.to_text() == "array = [1, 2] "

  attr_empty = AttributeNode(name="unit", value="")
  assert attr_empty.to_text() == "unit = "

  alias_str = AttributeAliasDefNode(
    leading_trivia=[Trivia(" ")],
    name="#map",
    value_str="affine_map<(d0) -> (d0)>",
    trailing_trivia=[Trivia(_NL)],
  )
  assert alias_str.to_text() == f" #map = affine_map<(d0) -> (d0)>{_NL}"

  alias_node = AttributeAliasDefNode(name="#map2", value_node=TypeNode(body="f32"))
  assert alias_node.to_text() == "#map2 = f32"

  type_alias = TypeAliasDefNode(name="my_alias", type_node=TypeNode(body=" i32 "))
  assert "!my_alias =" in type_alias.to_text()
  assert "i32" in type_alias.to_text()


def test_block_region_and_module_nodes() -> None:
  """Verify `BlockNode`, `RegionNode`, and `ModuleNode` formatting across empty and populated shapes."""
  empty_blk = BlockNode(label="")
  assert empty_blk.to_text() == ""
  assert BlockNode(label="^empty").to_text() == f"^empty:{_NL}"

  labeled_blk = BlockNode(
    leading_trivia=[Trivia("  ")],
    label="^bb0",
    arguments=[(ValueNode(name="%arg0"), TypeNode(body="i32"))],
    operations=[OperationNode(name="op", results=[ValueNode(name="%0")])],
    trailing_trivia=[Trivia(" ")],
  )
  blk_txt = labeled_blk.to_text()
  assert blk_txt.startswith("  ^bb0(%arg0: i32):")
  assert "%0 = op" in blk_txt

  reg = RegionNode(leading_trivia=[Trivia(" ")], blocks=[empty_blk], trailing_trivia=[Trivia(" ")])
  assert reg.to_text() == " { }"

  mod = ModuleNode(
    leading_trivia=[Trivia(" ")],
    aliases=[AttributeAliasDefNode(name="#a", value_str="1", trailing_trivia=[Trivia(_NL)])],
    body=BlockNode(label="", operations=[OperationNode(name="func.return")]),
    trailing_trivia=[Trivia(" ")],
  )
  mod_txt = mod.to_text()
  assert f"#a = 1{_NL}" in mod_txt
  assert "func.return" in mod_txt


def test_operation_node_formatting_variants() -> None:
  """Verify `OperationNode.to_text()` across generic ops, parens, successors, properties, tails, and trivia."""
  v0 = ValueNode(name="%0")
  v_space = ValueNode(name="%1", leading_trivia=[Trivia(" ")])
  t_i32 = TypeNode(body="i32")
  t_f32 = TypeNode(body="f32")
  attr = AttributeNode(name="a", value="1")

  # String and None `name_trivia` normalization
  op_str_trivia = OperationNode(name="foo", name_trivia=" ")
  assert isinstance(op_str_trivia.name_trivia, list)
  assert op_str_trivia.name_trivia[0].text == " "
  assert OperationNode(name="foo", name_trivia=None).name_trivia == []

  # Paren-less operands with and without leading trivia
  assert OperationNode(name="test.op", operands=[v_space], has_parens=False).to_text() == "test.op %1"
  assert OperationNode(name="foo", operands=[v0, ValueNode(name="%2")], has_parens=False).to_text() == "foo %0, %2"
  assert (
    OperationNode(name="foo", operands=[v0, v_space], has_parens=True, name_trivia=[Trivia(" ")]).to_text()
    == "foo (%0, %1)"
  )
  assert OperationNode(name="foo", operands=[v0], has_parens=False, name_trivia=[Trivia(" ")]).to_text() == "foo %0"

  # Multiple results, attributes, regions, and result_types tail variants
  assert "%0, %1 = sw.op" in OperationNode(name="sw.op", results=[v0, ValueNode(name="%1")]).to_text()
  assert OperationNode(name="sw.op", operands=[v0], attributes=[attr], name_trivia=[]).to_text() == "sw.op (%0) {a = 1}"
  assert OperationNode(name="foo", attributes=[attr], name_trivia=[Trivia(" ")]).to_text() == "foo {a = 1}"
  assert OperationNode(name="foo", attributes=[attr], name_trivia=[]).to_text() == "foo {a = 1}"
  assert OperationNode(name="foo", result_types=[t_i32]).to_text() == "foo : i32"
  assert OperationNode(name="foo", name_trivia=[Trivia(" ")], result_types=[t_i32]).to_text() == "foo : i32"
  assert OperationNode(name="foo", result_types=[t_i32, t_f32]).to_text() == "foo : (i32, f32)"
  assert (
    OperationNode(name="foo", result_types=[t_i32], op_tail_str=" -> ", op_tail_trivia=[Trivia(" ")]).to_text()
    == "foo  -> i32"
  )
  assert (
    OperationNode(name="foo", result_types=[t_i32, t_f32], op_tail_str="-> ", name_trivia="").to_text()
    == "foo-> (i32, f32)"
  )
  assert (
    OperationNode(name="foo", result_types=[t_i32, t_f32], op_tail_str=" : ", name_trivia="").to_text()
    == "foo : i32, f32"
  )
  assert (
    OperationNode(name="foo", result_types=[t_i32], op_tail_str="", op_tail_trivia=[], name_trivia=[]).to_text()
    == "foo : i32"
  )

  # Generic form, successors, properties, regions, and location with both empty and spaced tails
  generic_op = OperationNode(
    leading_trivia=[Trivia(" ")],
    name="my.op",
    is_generic=True,
    operands=[],
    successors=["^bb1", "^bb2"],
    properties=[AttributeNode("p", "1")],
    regions=[RegionNode(blocks=[BlockNode(label="^bb0", operations=[OperationNode(name="yield")])])],
    location="foo.py:1:1",
    trailing_trivia=[Trivia(_NL)],
  )
  gtxt = generic_op.to_text()
  assert '"my.op"' in gtxt
  assert "[^bb1, ^bb2]" in gtxt
  assert "<{p = 1}>" in gtxt
  assert "^bb0:" in gtxt
  assert "loc(foo.py:1:1)" in gtxt

  spaced_op = OperationNode(
    name="my.op",
    name_trivia=[Trivia(" ")],
    successors=["^bb1"],
    properties=[AttributeNode(name="x", value="1")],
  )
  assert "[^bb1] <{x = 1}>" in spaced_op.to_text()

  spaced_props_only = OperationNode(
    name="my.op",
    name_trivia=[Trivia(" ")],
    properties=[AttributeNode(name="x", value="1")],
  )
  assert "<{x = 1}>" in spaced_props_only.to_text()


def test_stablehlo_constant_op_variants() -> None:
  """Verify `StableHloConstantOp.to_text()` across custom, generic, multi-type, and attribute-free forms."""
  t_f32 = TypeNode(body="tensor<f32>")
  t_i32 = TypeNode(body="tensor<i32>")
  val_attr = AttributeNode(name="value", value="dense<1.0>")

  op1 = StableHloConstantOp(name="stablehlo.constant", attributes=[val_attr])
  assert op1.to_text() == "stablehlo.constant dense<1.0>"

  op2 = StableHloConstantOp(
    name="stablehlo.constant",
    results=[ValueNode(name="%0")],
    name_trivia=[Trivia("  ")],
    attributes=[val_attr],
    result_types=[t_f32, t_i32],
    leading_trivia=[Trivia("  ")],
    trailing_trivia=[Trivia(_NL)],
  )
  assert op2.to_text() == f"  %0 = stablehlo.constant  dense<1.0> : (tensor<f32>, tensor<i32>){_NL}"

  op3 = StableHloConstantOp(name="stablehlo.constant", attributes=[val_attr], result_types=[t_f32])
  assert op3.to_text() == "stablehlo.constant dense<1.0> : tensor<f32>"

  op4 = StableHloConstantOp(name="stablehlo.constant", results=[ValueNode(name="%0")], result_types=[t_f32, t_i32])
  assert ": (tensor<f32>, tensor<i32>)" in op4.to_text()

  op5 = StableHloConstantOp(name="stablehlo.constant")
  assert op5.to_text() == "stablehlo.constant"

  generic_const = StableHloConstantOp(
    name='"my.op"',
    operands=[],
    successors=["^bb1"],
    properties=[AttributeNode("p", "1")],
    location="foo.py:1:1",
  )
  gconst_txt = generic_const.to_text()
  assert "[^bb1]" in gconst_txt
  assert "<{p = 1}>" in gconst_txt
  assert "loc(foo.py:1:1)" in gconst_txt
