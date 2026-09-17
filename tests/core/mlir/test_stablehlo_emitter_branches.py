"""Test module."""

import typing

import libcst as cst

from ml_switcheroo.core.mlir.cst import OperationNode
from ml_switcheroo.core.mlir.stablehlo_emitter import StableHloEmitter
from ml_switcheroo.semantics.manager import SemanticsManager


def test_stablehlo_branches() -> None:
  """Docstring."""
  semantics = SemanticsManager()

  # Mock some behavior
  def mock_get_def(name: str) -> typing.Optional[tuple[str, dict[str, typing.Any]]]:
    """Mocks get_definition."""
    return ("dummy", {"variants": {}})

  semantics.get_definition = mock_get_def  # type: ignore

  emitter = StableHloEmitter(semantics)

  # 87->91 (if param.annotation is None)
  # 110->115 (elif infer_pass.return_types where it is empty)
  code: str = "def foo(x):\n    pass"
  tree: cst.Module = cst.parse_module(code)
  emitter.convert(tree)

  # 155->161 (if without else) -> tested in no_else? Wait, if getattr(node, 'orelse', None) is None.
  # We already have test_conditional_control_flow_no_else

  # 193->200 (return without value)
  code = "def foo():\n    return"
  tree = cst.parse_module(code)
  emitter.convert(tree)

  # 218->224 (op.name != sw.op and != sw.constant)
  # 215->232 (empty ops loop?) - it will happen with some expr
  code = "def foo():\n    x = 1"
  tree = cst.parse_module(code)
  emitter.convert(tree)

  # 248->253 (sw.constant without value attr)
  op = OperationNode(name="sw.constant", operands=[], attributes=[])
  emitter._resolve_sw_constant(op)

  # 278->280 (sw.op without type attr)
  op = OperationNode(name="sw.op", operands=[], attributes=[])
  emitter._resolve_sw_op(op)

  # 319->exit (already hit in my extra test? but we need generator or something?)
  # 331->exit
  # 357->exit

  # 407->410 (if not stablehlo_name -> already hit when it's unknown)

  # Let's try _emit_statement with something that gives empty ops or dummy import
  code = "import x\nfrom y import z"
  tree = cst.parse_module(code)
  emitter.convert(tree)


def test_stablehlo_if_orelse_invalid() -> None:
  """Hit the 209->225 branch by providing an invalid orelse."""
  semantics = SemanticsManager()
  emitter = StableHloEmitter(semantics)

  # Parse a normal if, then mutate its orelse to a bad type.
  tree = cst.parse_module("if True: pass\n")

  # We need to extract the If node.
  if_node = tree.body[0]
  if isinstance(if_node, cst.If):
    # We create a new If with bad orelse
    bad_if = if_node.with_changes(orelse=cst.Pass())
    # Hack around the AST validation by just passing it to _emit_if
    ops = emitter._emit_if(bad_if)
    # Assert ops generated
    assert len(ops) > 0


def test_stablehlo_additional_branches() -> None:
  """Test remaining branches in StableHloEmitter."""
  from unittest.mock import patch
  from ml_switcheroo.core.mlir.cst import AttributeNode, TypeNode

  semantics = SemanticsManager()
  emitter = StableHloEmitter(semantics)

  # 80->79: param with non-Name name
  fn = cst.parse_module("def foo(): pass").body[0]
  assert isinstance(fn, cst.FunctionDef)
  bad_param = typing.cast(cst.Param, cst.Param(name=typing.cast(cst.Name, cst.Integer("1"))))
  fn_bad_param = fn.with_deep_changes(fn.params, params=[bad_param])
  emitter._emit_func_def(fn_bad_param)

  # 156: while loop with empty body (pass emits no ops)
  code = "while True:\n    pass"
  tree = cst.parse_module(code)
  while_node = tree.body[0]
  assert isinstance(while_node, cst.While)
  ops = emitter._emit_while(while_node)
  assert len(ops) > 0

  # 204: if with else pass (false_block has no operations)
  code = "if True:\n    x = 1\nelse:\n    pass"
  tree = cst.parse_module(code)
  if_node = tree.body[0]
  assert isinstance(if_node, cst.If)
  ops = emitter._emit_if(if_node)
  assert len(ops) > 0

  # 206: if with else assignment (false_block has ops but not return)
  code = "if True:\n    x = 1\nelse:\n    x = 2"
  tree = cst.parse_module(code)
  if_node = tree.body[0]
  assert isinstance(if_node, cst.If)
  ops = emitter._emit_if(if_node)
  assert len(ops) > 0

  # 214: if with elif where nested _emit_if returns []
  elif_tree = cst.parse_module("if True: pass\nelif False: pass")
  elif_node = elif_tree.body[0]
  assert isinstance(elif_node, cst.If)
  calls = 0
  orig_emit_if = emitter._emit_if

  def fake_emit_if(node: typing.Any) -> typing.List[OperationNode]:
    """Mock recursive _emit_if call."""
    nonlocal calls
    calls += 1
    if calls > 1:
      return []
    return orig_emit_if(node)

  with patch.object(emitter, "_emit_if", side_effect=fake_emit_if):
    ops = emitter._emit_if(elif_node)
    assert len(ops) > 0

  # 215->217: if with elif where nested _emit_if returns [OperationNode(name="stablehlo.return")]
  calls_ret = 0

  def fake_emit_if_ret(node: typing.Any) -> typing.List[OperationNode]:
    """Mock recursive _emit_if call returning a return op."""
    nonlocal calls_ret
    calls_ret += 1
    if calls_ret > 1:
      return [OperationNode(name="stablehlo.return")]
    return orig_emit_if(node)

  with patch.object(emitter, "_emit_if", side_effect=fake_emit_if_ret):
    ops = emitter._emit_if(elif_node)
    assert len(ops) > 0

  # 324->exit: _resolve_sw_constant with result_types already set
  op_const = OperationNode(
    name="sw.constant",
    operands=[],
    attributes=[AttributeNode(name="value", value="1.0")],
    result_types=[TypeNode(body="tensor<f32>")],
  )
  emitter._resolve_sw_constant(op_const)
  assert len(op_const.result_types) == 1

  # 349->exit: _resolve_sw_op with result_types already set
  with patch.object(emitter, "_lookup_stablehlo_op", return_value="stablehlo.add"):
    op_sw = OperationNode(
      name="sw.op",
      operands=[],
      attributes=[AttributeNode(name="type", value='"add"')],
      result_types=[TypeNode(body="tensor<f32>")],
    )
    emitter._resolve_sw_op(op_sw)
    assert op_sw.name == "stablehlo.add"
    assert len(op_sw.result_types) == 1
