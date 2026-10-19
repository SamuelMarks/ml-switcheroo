"""Test suite for MLIR CST nodes (`src/ml_switcheroo/core/mlir/cst.py`)."""

from ml_switcheroo.core.mlir.cst import (
  AttributeAliasDefNode,
  AttributeNode,
  BlockNode,
  ModuleNode,
  OperationNode,
  RegionNode,
  StableHloConstantOp,
  Trivia,
  TriviaNode,
  TypeAliasDefNode,
  TypeNode,
  ValueNode,
)


def test_trivia_node() -> None:
  """Verify Trivia / TriviaNode construction and serialization."""
  n: TriviaNode = TriviaNode("  ")
  assert n.to_text() == "  "
  assert n.content == "  "

  n_kw: Trivia = Trivia(content="// comment", kind="comment")
  assert n_kw.text == "// comment"
  assert n_kw.kind == "comment"
  assert n_kw.to_text() == "// comment"


def test_value_node() -> None:
  """Verify ValueNode construction and serialization."""
  n: ValueNode = ValueNode("%0")
  assert n.to_text() == "%0"

  n_trivia: ValueNode = ValueNode("%0", leading_trivia=[TriviaNode(" ")], trailing_trivia=[TriviaNode(" ")])
  assert n_trivia.to_text() == " %0 "


def test_type_node() -> None:
  """Verify TypeNode construction and serialization."""
  n: TypeNode = TypeNode("!sw.type")
  assert n.to_text() == "!sw.type"

  n_trivia: TypeNode = TypeNode("i32", leading_trivia=[TriviaNode(" ")], trailing_trivia=[TriviaNode(" ")])
  assert n_trivia.to_text() == " i32 "


def test_attribute_node() -> None:
  """Verify AttributeNode construction and serialization."""
  n_str: AttributeNode = AttributeNode(name="sym_name", value='"my_func"')
  assert n_str.to_text() == 'sym_name = "my_func"'

  n_list: AttributeNode = AttributeNode(name="bases", value=['"Base1"', '"Base2"'])
  assert n_list.to_text() == 'bases = ["Base1", "Base2"]'

  n_type: AttributeNode = AttributeNode(name="value", value="42", type_annotation="i32")
  assert n_type.to_text() == "value = 42 : i32"

  n_trivia: AttributeNode = AttributeNode(
    name="name", value='"val"', leading_trivia=[TriviaNode(" ")], trailing_trivia=[TriviaNode(" ")]
  )
  assert n_trivia.to_text() == ' name = "val" '


def test_block_node() -> None:
  """Verify BlockNode construction and serialization."""
  n_empty: BlockNode = BlockNode(label="^bb0")
  assert n_empty.to_text().strip() == "^bb0:"

  arg_val: ValueNode = ValueNode("%arg0")
  arg_typ: TypeNode = TypeNode("i32")
  n_args: BlockNode = BlockNode(label="^bb0", arguments=[(arg_val, arg_typ)])
  assert n_args.to_text().strip() == "^bb0(%arg0: i32):"

  op: OperationNode = OperationNode(name="sw.return", trailing_trivia=[TriviaNode(" ")])
  n_ops: BlockNode = BlockNode(label="", operations=[op])
  assert n_ops.to_text() == "sw.return "

  n_trivia: BlockNode = BlockNode(label="^bb0", leading_trivia=[TriviaNode("  ")])
  assert n_trivia.to_text().startswith("  ^bb0:")


def test_region_node() -> None:
  """Verify RegionNode construction and serialization."""
  b: BlockNode = BlockNode(label="^bb0")
  n: RegionNode = RegionNode(blocks=[b])
  assert n.to_text().startswith("{^bb0:")

  n_trivia: RegionNode = RegionNode(blocks=[], leading_trivia=[TriviaNode(" ")], trailing_trivia=[TriviaNode(" ")])
  assert n_trivia.to_text() == " { }"


def test_operation_node() -> None:
  """Verify OperationNode construction and serialization."""
  n: OperationNode = OperationNode(name="sw.return")
  assert n.to_text() == "sw.return"

  n_res_op: OperationNode = OperationNode(
    name="sw.add", results=[ValueNode("%res")], operands=[ValueNode("%a"), ValueNode("%b")]
  )
  assert n_res_op.to_text() == "%res = sw.add (%a, %b)"

  n_attr_type: OperationNode = OperationNode(
    name="sw.constant", attributes=[AttributeNode(name="value", value="42")], result_types=[TypeNode("i32")]
  )
  assert n_attr_type.to_text() == "sw.constant {value = 42} : i32"

  n_multi_type: OperationNode = OperationNode(name="sw.div", result_types=[TypeNode("i32"), TypeNode("i32")])
  assert n_multi_type.to_text() == "sw.div : (i32, i32)"

  r: RegionNode = RegionNode(blocks=[])
  n_region: OperationNode = OperationNode(name="sw.func", regions=[r])
  assert n_region.to_text() == "sw.func {}"

  n_trivia: OperationNode = OperationNode(
    name="sw.test",
    leading_trivia=[TriviaNode("// l ")],
    name_trivia=[TriviaNode(" ")],
    trailing_trivia=[TriviaNode("// t")],
  )
  assert n_trivia.to_text() == "// l sw.test // t"


def test_stablehlo_constant_op() -> None:
  """Verify StableHloConstantOp construction and serialization."""
  n: StableHloConstantOp = StableHloConstantOp(
    name="stablehlo.constant",
    results=[ValueNode("%0")],
    attributes=[AttributeNode(name="value", value="dense<1.0>")],
    result_types=[TypeNode("tensor<f32>")],
  )
  assert n.to_text() == "%0 = stablehlo.constant dense<1.0> : tensor<f32>"

  n_multi: StableHloConstantOp = StableHloConstantOp(
    name="stablehlo.constant",
    attributes=[AttributeNode(name="value", value="dense<1>")],
    result_types=[TypeNode("i32"), TypeNode("i64")],
  )
  assert n_multi.to_text() == "stablehlo.constant dense<1> : (i32, i64)"

  n_trivia: StableHloConstantOp = StableHloConstantOp(
    name="stablehlo.constant",
    name_trivia=[TriviaNode(" ")],
    attributes=[AttributeNode(name="value", value="1")],
    trailing_trivia=[TriviaNode(" // a")],
  )
  assert n_trivia.to_text() == "stablehlo.constant 1 // a"


def test_module_and_alias_nodes() -> None:
  """Verify ModuleNode, AttributeAliasDefNode, and TypeAliasDefNode serialization."""
  t_alias = TypeAliasDefNode(name="my_int", type_node=TypeNode(" i32 "))
  a_alias = AttributeAliasDefNode(name="#map", value_str="affine_map<(d0) -> (d0)> ")
  b: BlockNode = BlockNode(label="^bb0")
  m: ModuleNode = ModuleNode(
    body=b,
    aliases=[t_alias, a_alias],
    leading_trivia=[TriviaNode("// top ")],
    trailing_trivia=[TriviaNode(" // bot")],
  )
  rendered = m.to_text()
  assert "!my_int = i32" in rendered
  assert "#map = affine_map<(d0) -> (d0)>" in rendered
  assert "// top " in rendered
  assert "^bb0:" in rendered
  assert " // bot" in rendered
