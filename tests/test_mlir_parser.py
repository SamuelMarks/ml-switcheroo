"""Test suite for the MLIR CST Parser and Transformer."""

import typing
import pytest
from lark import Token, Tree

from ml_switcheroo.core.cst.base import Trivia
from ml_switcheroo.core.mlir.cst import (
  AttributeNode,
  BlockNode,
  ModuleNode,
  OperationNode,
  RegionNode,
)
from ml_switcheroo.core.mlir.parser import (
  MlirLexer,
  MlirParser,
  MlirToken,
  MlirTransformer,
  _get_trivia,
)


def test_parser_comprehensive() -> None:
  """Test parsing a comprehensive custom operation."""
  mlir: str = """
    %res1, %res2 = "dialect.op1" @symbol (%arg0 : !type1, %arg1 : !type2)
    """
  parser: MlirParser = MlirParser(mlir)
  node: ModuleNode = parser.parse()
  assert node is not None


def test_parse_empty() -> None:
  """Test parsing empty string returns an empty ModuleNode."""
  parser: MlirParser = MlirParser("")
  node: ModuleNode = parser.parse()
  assert len(node.body.operations) == 0


def test_parse_unexpected_token() -> None:
  """Test that illegal tokens raise a ValueError with unexpected token message."""
  parser: MlirParser = MlirParser("~")
  with pytest.raises(ValueError, match="Unexpected"):
    parser.parse()


def test_attribute_alias() -> None:
  """Test parsing attribute aliases with various values."""
  mlir: str = """
    #map0 = "some_string"
    #arr = [1, 2]
    module {
    }
    """
  parser: MlirParser = MlirParser(mlir)
  node: ModuleNode = parser.parse()
  assert len(node.aliases) == 2
  assert node.aliases[0].name == "#map0"
  assert node.aliases[1].name == "#arr"


def test_operation_complex() -> None:
  """Test parsing complex operations including blocks, regions, and attributes."""
  mlir: str = """
    %0 = "foo.bar" (%1, %2) { attr1 = "val1", attr2 = [1, 2, 3] } {
      ^bb0(%arg0: i32, %arg1: f32):
        "foo.yield"() : () -> ()
    } : (i32, i32) -> i32

    "foo.return" %0 : i32

    sw.add %0, %1 : i32
    """
  parser: MlirParser = MlirParser(mlir)
  node: ModuleNode = parser.parse()
  assert len(node.body.operations) == 3


def test_lexer_state_object() -> None:
  """Test MlirLexer tokenizing with an object having a text attribute."""

  class LexerStateMock:
    """Mock lexer state containing text."""

    def __init__(self, text: str) -> None:
      """Initialize mock state with text."""
      self.text: str = text

  lexer: MlirLexer = MlirLexer(None)
  tokens: list[Token] = list(lexer.lex(LexerStateMock("%x = sw.op")))
  assert len(tokens) > 0


def test_get_trivia_empty() -> None:
  """Test _get_trivia returns an empty list when node has no trivia and no children."""
  assert _get_trivia(object()) == []

  class EmptyTree:
    """Class with empty children list."""

    children: list[typing.Any] = []

  assert _get_trivia(EmptyTree()) == []


def test_type_alias_def() -> None:
  """Test parsing MLIR type alias definitions."""
  mlir: str = "!my_type = !sw.type<f32>"
  parser: MlirParser = MlirParser(mlir)
  node: ModuleNode = parser.parse()
  assert len(node.aliases) == 1
  alias = node.aliases[0]
  from ml_switcheroo.core.mlir.cst import TypeAliasDefNode

  assert isinstance(alias, TypeAliasDefNode)
  assert alias.name == "my_type"
  assert alias.type_node is not None
  assert alias.type_node.body == "!sw.type<f32>"


def test_type_alias_def_trailing_trivia() -> None:
  """Test type_alias_def transformer with trailing trivia."""
  tok_name: MlirToken = MlirToken("TYPE", "!t")
  tok_name.leading_trivia = []
  name_child: Tree[MlirToken] = Tree("type_alias", [tok_name])
  eq_tok: MlirToken = MlirToken("EQ", "=")
  eq_tok.leading_trivia = []
  type_tok: MlirToken = MlirToken("TYPE", "!other")
  type_tok.leading_trivia = []
  triv_tok: MlirToken = MlirToken("WS", " ")
  triv_tok.leading_trivia = [Trivia(" ")]
  triv_tree: Tree[MlirToken] = Tree("trivia", [triv_tok])
  transformer: MlirTransformer = MlirTransformer()
  res = transformer.type_alias_def([name_child, eq_tok, type_tok, triv_tree])
  assert res.name == "t"
  assert len(res.trailing_trivia) == 1


def test_attribute_alias_def_fallback_and_trivia() -> None:
  """Test attribute_alias_def transformer with non-value child and trailing trivia."""
  name_tok: MlirToken = MlirToken("ATTR_ALIAS_ID", "#foo")
  name_tok.leading_trivia = []
  name_tree: Tree[MlirToken] = Tree("attribute_alias", [name_tok])
  eq_tok: MlirToken = MlirToken("EQ", "=")
  eq_tok.leading_trivia = []
  val_child: Tree[MlirToken] = Tree("custom_val", [])
  val_node: Tree[typing.Any] = Tree("attribute_value", [val_child])
  triv_tok: MlirToken = MlirToken("WS", " ")
  triv_tok.leading_trivia = [Trivia(" ")]
  triv_tree: Tree[MlirToken] = Tree("trivia", [triv_tok])
  transformer: MlirTransformer = MlirTransformer()
  res = transformer.attribute_alias_def([name_tree, eq_tok, val_node, triv_tree])
  assert res.name == "#foo"
  assert len(res.trailing_trivia) == 1


def test_operation_none_children() -> None:
  """Test operation transformer with empty or None children."""
  transformer: MlirTransformer = MlirTransformer()
  op: OperationNode = transformer.operation([None])
  assert op.leading_trivia == []


def test_generic_operation_with_results() -> None:
  """Test generic operation with results and string generic name."""
  mlir: str = '%0:1, %1:1 = "generic.test"()'
  parser: MlirParser = MlirParser(mlir)
  node: ModuleNode = parser.parse()
  assert len(node.body.operations) == 1
  op: OperationNode = node.body.operations[0]
  assert op.is_generic is True
  assert op.name == "generic.test"
  assert len(op.results) == 2


def test_operands_variations() -> None:
  """Test operand with symbol reference and use_index."""
  mlir: str = "sw.op(%0#1, @symbol : !type)"
  parser: MlirParser = MlirParser(mlir)
  node: ModuleNode = parser.parse()
  op: OperationNode = node.body.operations[0]
  assert op.operands[0].use_index == 1
  assert op.operands[1].name == "@symbol"


def test_generic_operation_value_use_list() -> None:
  """Test generic operation with operands in value_use_list."""
  mlir: str = '%0:1 = "generic.add"(%a, %b)'
  parser: MlirParser = MlirParser(mlir)
  node: ModuleNode = parser.parse()
  op: OperationNode = node.body.operations[0]
  assert len(op.operands) == 2
  assert op.operands[0].name == "%a"
  assert op.operands[1].name == "%b"


def test_generic_operation_successors() -> None:
  """Test generic operation with successors."""
  mlir: str = '%0:1 = "generic.br"()[^bb1, ^bb2]'
  parser: MlirParser = MlirParser(mlir)
  node: ModuleNode = parser.parse()
  op: OperationNode = node.body.operations[0]
  assert op.successors == ["^bb1", "^bb2"]


def test_operation_caret_id_tree() -> None:
  """Test operation transformer with caret_id Tree child in successor."""
  caret_tree: Tree[Token] = Tree("caret_id", [Token("CARET", "^"), Token("SUFFIX", "bb3")])
  succ_tree: Tree[Token] = Tree("successor", [caret_tree])
  succ_list: Tree[Token] = Tree("successor_list", [succ_tree])
  transformer: MlirTransformer = MlirTransformer()
  op: OperationNode = transformer.operation([succ_list])
  assert op.successors == ["^bb3"]

  # Also test with non-caret token in successor
  succ_tree_other: Tree[Token] = Tree("successor", [Token("OTHER", "dummy")])
  succ_list_other: Tree[Token] = Tree("successor_list", [succ_tree_other])
  op_other: OperationNode = transformer.operation([succ_list_other])
  assert op_other.successors == []


def test_operation_region_list_tree() -> None:
  """Test operation transformer with region_list Tree."""
  reg_tree: Tree[typing.Any] = Tree("region", [BlockNode(label="bb0")])
  reg_list: Tree[typing.Any] = Tree("region_list", [Token("LPAREN", "("), reg_tree])
  transformer: MlirTransformer = MlirTransformer()
  op: OperationNode = transformer.operation([reg_list])
  assert len(op.regions) == 1


def test_generic_operation_properties() -> None:
  """Test generic operation with dictionary properties."""
  mlir: str = '%0:1 = "generic.op"() <{ prop = "val" }>'
  parser: MlirParser = MlirParser(mlir)
  node: ModuleNode = parser.parse()
  op: OperationNode = node.body.operations[0]
  assert len(op.properties) == 1
  assert op.properties[0].name == "prop"


def test_generic_operation_function_types() -> None:
  """Test generic operation with function type variations."""
  mlir1: str = '%0:1 = "generic.multi"(%a, %b) : (i32, f32) -> (i32, f32)'
  parser1: MlirParser = MlirParser(mlir1)
  op1: OperationNode = parser1.parse().body.operations[0]
  assert len(op1.result_types) == 2
  assert op1.operands[0].type_node is not None
  assert op1.operands[1].type_node is not None

  mlir2: str = '%0:1 = "generic.single"(%a) : i32 -> i32'
  parser2: MlirParser = MlirParser(mlir2)
  op2: OperationNode = parser2.parse().body.operations[0]
  assert len(op2.result_types) == 1
  assert op2.operands[0].type_node is not None

  mlir3: str = '%0:1 = "generic.mix"(%a) : i32 -> (i32, f32)'
  parser3: MlirParser = MlirParser(mlir3)
  op3: OperationNode = parser3.parse().body.operations[0]
  assert len(op3.result_types) == 2

  mlir4: str = '%0:1 = "generic.mix2"(%a, %b) : (i32, f32) -> i32'
  parser4: MlirParser = MlirParser(mlir4)
  op4: OperationNode = parser4.parse().body.operations[0]
  assert len(op4.result_types) == 1

  # Also test when operands count is less than arg types
  mlir5: str = '%0:1 = "generic.extra_types"() : (i32, f32) -> i32'
  parser5: MlirParser = MlirParser(mlir5)
  op5: OperationNode = parser5.parse().body.operations[0]
  assert len(op5.result_types) == 1

  mlir6: str = '%0:1 = "generic.extra_types2"() : i32 -> i32'
  parser6: MlirParser = MlirParser(mlir6)
  op6: OperationNode = parser6.parse().body.operations[0]
  assert len(op6.result_types) == 1


def test_operation_function_type_branches() -> None:
  """Test operation transformer with non-type parts before and after arrow."""
  transformer: MlirTransformer = MlirTransformer()
  fn_tree: Tree[Token] = Tree(
    "function_type",
    [Token("OTHER", "x"), Token("ARROW", "->"), Token("OTHER", "y")],
  )
  op: OperationNode = transformer.operation([fn_tree])
  assert op.result_types == []


def test_trailing_location() -> None:
  """Test operation with trailing location."""
  mlir: str = 'sw.op() loc("my_file.py:10:2")'
  parser: MlirParser = MlirParser(mlir)
  node: ModuleNode = parser.parse()
  op: OperationNode = node.body.operations[0]
  assert op.location == '"my_file.py:10:2"'


def test_operation_empty_op_tail() -> None:
  """Test operation transformer with empty op_tail."""
  tail_tree: Tree[typing.Any] = Tree("op_tail", [])
  transformer: MlirTransformer = MlirTransformer()
  op: OperationNode = transformer.operation([tail_tree])
  assert op.op_tail_str == ""


def test_dictionary_attribute_dialect_attr() -> None:
  """Test dictionary attribute with dialect attribute syntax."""
  mlir: str = "sw.op() { my_attr = #alias.ident }"
  parser: MlirParser = MlirParser(mlir)
  node: ModuleNode = parser.parse()
  op: OperationNode = node.body.operations[0]
  assert op.attributes[0].name == "my_attr"
  assert op.attributes[0].value == "#alias.ident"


def test_block_non_op_child() -> None:
  """Test block transformer with arbitrary child."""
  transformer: MlirTransformer = MlirTransformer()
  blk: BlockNode = transformer.block(["ignored_string", OperationNode(name="op")])
  assert len(blk.operations) == 1


def test_regions_transformer() -> None:
  """Test regions transformer filtering RegionNodes."""
  transformer: MlirTransformer = MlirTransformer()
  r: RegionNode = RegionNode()
  res: list[RegionNode] = transformer.regions([r, "dummy"])
  assert res == [r]


def test_module_transformer_empty() -> None:
  """Test module transformer when given empty list of children."""
  transformer: MlirTransformer = MlirTransformer()
  mod: ModuleNode = transformer.module([])
  assert mod.leading_trivia == []


def test_operation_raw_lists() -> None:
  """Test operation transformer handling raw list of RegionNode and AttributeNode."""
  transformer: MlirTransformer = MlirTransformer()
  attr: AttributeNode = AttributeNode(name="attr")
  reg: RegionNode = RegionNode()
  op1: OperationNode = transformer.operation([[attr]])
  assert op1.attributes == [attr]
  op2: OperationNode = transformer.operation([[reg]])
  assert op2.regions == [reg]
  op3: OperationNode = transformer.operation([[]])
  assert op3.attributes == []
  op4: OperationNode = transformer.operation([["other"]])
  assert op4.attributes == []


def test_operation_operands_empty() -> None:
  """Test operation transformer with operands Tree with no children."""
  transformer: MlirTransformer = MlirTransformer()
  operands_tree: Tree[typing.Any] = Tree("operands", [])
  op: OperationNode = transformer.operation([operands_tree])
  assert op.has_parens is False


def test_operation_function_type_no_arrow() -> None:
  """Test operation transformer with function_type tree without ARROW."""
  transformer: MlirTransformer = MlirTransformer()
  fn_type_tree: Tree[typing.Any] = Tree("function_type", [Token("TYPE", "i32")])
  op: OperationNode = transformer.operation([fn_type_tree])
  assert op.result_types == []


def test_block_empty_children() -> None:
  """Test block transformer with empty children list."""
  transformer: MlirTransformer = MlirTransformer()
  blk: BlockNode = transformer.block([])
  assert blk.leading_trivia == []
