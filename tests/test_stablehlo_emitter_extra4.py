"""Module."""

import libcst as cst
from unittest import mock
from ml_switcheroo.core.mlir.stablehlo_emitter import StableHloEmitter
from ml_switcheroo.core.mlir.cst import OperationNode, AttributeNode, TypeNode


class MockTypeInference(cst.CSTVisitor):
  """Class doc."""

  def __init__(self, *args, **kwargs):
    """Init doc."""
    self.return_types = []
    self.env = {}

  def visit(self, node):
    """Function doc."""
    pass


def test_param_not_name():
  """Function doc."""
  with mock.patch("ml_switcheroo.core.mlir.stablehlo_emitter.TypeInferencePass", MockTypeInference):
    emitter = StableHloEmitter(semantics=mock.Mock())
    code = """
def f(x):
    pass
"""
    mod = cst.parse_module(code)
    # Modify the param to not be a name
    func = mod.body[0]
    # We need to create a Param with a name that is not cst.Name
    # But cst.Param.name must be a BaseExpression. Let's make it a cst.Attribute.
    new_param = func.params.params[0].with_changes(name=cst.Attribute(value=cst.Name("a"), attr=cst.Name("b")))
    new_params = func.params.with_changes(params=(new_param,))
    new_func = func.with_changes(params=new_params)

    # It should just skip it and not crash.
    emitter._emit_func_def(new_func)


def test_if_false_branch_no_return():
  """Function doc."""
  with mock.patch("ml_switcheroo.core.mlir.stablehlo_emitter.TypeInferencePass", MockTypeInference):
    emitter = StableHloEmitter(semantics=mock.Mock())
    # Else branch has an assignment (no return)
    code = """
if cond:
    return 1
else:
    x = 2
"""
    mod = cst.parse_module(code)
    if_stmt = mod.body[0]
    ops = emitter._emit_if(if_stmt)
    assert len(ops) > 0
    if_op = ops[-1]
    assert if_op.name == "stablehlo.if"
    false_region = if_op.regions[1]
    # the last op should be stablehlo.return
    assert false_region.blocks[0].operations[-1].name == "stablehlo.return"


def test_if_elif_no_return():
  """Function doc."""
  with mock.patch("ml_switcheroo.core.mlir.stablehlo_emitter.TypeInferencePass", MockTypeInference):
    emitter = StableHloEmitter(semantics=mock.Mock())
    # Elif branch has an assignment (no return)
    code = """
if cond:
    return 1
elif cond2:
    x = 2
"""
    mod = cst.parse_module(code)
    if_stmt = mod.body[0]
    ops = emitter._emit_if(if_stmt)
    assert len(ops) > 0
    if_op = ops[-1]
    assert if_op.name == "stablehlo.if"
    false_region = if_op.regions[1]  # This is the elif block which has a stablehlo.if inside it, wait.
    # Inside the false_region (which contains the elif's if), the ops are for the nested if.
    # But _emit_if returns the ops for the nested if.
    # If the nested if doesn't end with a return, _emit_if will append one!
    # wait, let's check stablehlo_emitter logic:
    # false_block = BlockNode(label="", operations=self._emit_if(node.orelse))
    # if not false_block.operations ... elif false_block.operations[-1].name not in ... append return
    # so it will append a return AFTER the nested if_op in the false_block.
    assert false_region.blocks[0].operations[-1].name == "stablehlo.return"


def test_resolve_sw_constant_with_result_type():
  """Function doc."""
  emitter = StableHloEmitter(semantics=mock.Mock())
  op = OperationNode(
    name="sw.constant",
    attributes=[AttributeNode(name="value", value='"42"')],
    result_types=[TypeNode(body="tensor<i64>")],
  )
  emitter._resolve_sw_constant(op)
  # result_types should not be overridden
  assert op.result_types[0].body == "tensor<i64>"


def test_resolve_sw_op_with_result_type():
  """Function doc."""
  emitter = StableHloEmitter(semantics=mock.Mock())
  op = OperationNode(
    name="sw.op", attributes=[AttributeNode(name="type", value='"np.add"')], result_types=[TypeNode(body="tensor<i64>")]
  )
  with mock.patch.object(emitter, "_lookup_stablehlo_op", return_value="stablehlo.add"):
    emitter._resolve_sw_op(op)
  # result_types should not be overridden
  assert op.result_types[0].body == "tensor<i64>"


def test_if_elif_empty():
  """Function doc."""
  with mock.patch("ml_switcheroo.core.mlir.stablehlo_emitter.TypeInferencePass", MockTypeInference):
    emitter = StableHloEmitter(semantics=mock.Mock())
    code = """
if cond:
    pass
elif cond2:
    pass
"""
    mod = cst.parse_module(code)
    if_stmt = mod.body[0]
    # mock _emit_if to return empty list when called for the elif
    original_emit_if = emitter._emit_if

    def side_effect(node):
      """Function doc."""
      if isinstance(node, cst.If) and node.test.value == "cond2":
        return []
      return original_emit_if(node)

    with mock.patch.object(emitter, "_emit_if", side_effect=side_effect):
      ops = emitter._emit_if(if_stmt)
      if_op = ops[-1]
      false_region = if_op.regions[1]
      assert false_region.blocks[0].operations[-1].name == "stablehlo.return"
