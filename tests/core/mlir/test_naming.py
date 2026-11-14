"""Unit tests for MLIR SSA-to-Python `NamingContext` and generator naming."""

from ml_switcheroo.core.mlir.cst import AttributeNode, BlockNode, ModuleNode, OperationNode, ValueNode
from ml_switcheroo.core.mlir.generator import MlirToPythonGenerator
from ml_switcheroo.core.mlir.naming import NamingContext


def _gen_code(ops: list[OperationNode]) -> str:
  """Lower a list of MLIR `OperationNode` instances to Python source code.

  Args:
      ops: Top-level MLIR operations to convert.

  Returns:
      Generated Python module source string.
  """
  mod = ModuleNode(body=BlockNode(label="", operations=ops))
  return MlirToPythonGenerator().generate(mod).code


def test_naming_context_initialization_and_lookup() -> None:
  """Verify `NamingContext` default state, registration, symbol `@` stripping, and fallback lookup."""
  ctx = NamingContext()
  assert ctx._map == {}
  assert ctx._used_names == {}
  assert "return" in ctx._reserved
  assert "self" not in ctx._reserved

  assert ctx.register("%0") == "_0"
  assert ctx.register("%val", hint="val") == "val"
  assert ctx.lookup("%val") == "val"
  assert ctx.lookup("@my_func") == "my_func"
  assert ctx.lookup("%unregistered") == "_unregistered"
  assert ctx.lookup("unknown") == "unknown"


def test_naming_context_hints_keywords_and_collisions() -> None:
  """Verify `NamingContext.register` handles digit stripping, reserved keywords, collisions, and invalid hints."""
  ctx = NamingContext()
  assert ctx.register("%self0", hint="self0") == "self"
  assert ctx.register("%x1", hint="self1") == "_self1"
  assert ctx.register("%return1", hint="return1") == "return1"
  assert ctx.register("%return", hint="return") == "_return"
  assert ctx.register("%1", hint="flatten") == "_flatten"
  assert ctx.register("%out_abc", hint="_semantic") == "_semantic"
  assert ctx.register("%bad", hint="1bad") == "_1bad"
  assert ctx.register("%2", hint="invalid-name") == "_invalid-name_0"
  assert ctx.register("%other", hint=".b") == "_b"
  assert ctx.register("%empty_hint", hint="") == "_empty_hint"

  # Reserved MLIR SSA keyword with collision loop
  ctx2 = NamingContext()
  assert ctx2.register("%class", hint="%class") == "_class"
  ctx2._used_names["class"] = True
  ctx2._used_names["_class_0"] = True
  assert ctx2.register("%class_next", hint="%class") == "_class_1"

  # Direct name collision, underscore-prefixed collision, and non-`%` fallback loop
  ctx3 = NamingContext()
  ctx3._used_names["coll"] = True
  assert ctx3.register("%coll", hint="coll") == "_coll"

  ctx3._used_names["coll2"] = True
  ctx3._used_names["_coll2"] = True
  assert ctx3.register("%coll2", hint="coll2") == "_coll2_0"

  ctx3._used_names["_already"] = True
  ctx3._used_names["_already_0"] = True
  assert ctx3.register("%out4", hint="_already") == "_already_1"

  ctx3._used_names["_ssa2"] = True
  ctx3._used_names["_ssa2_0"] = True
  assert ctx3.register("%ssa2") == "_ssa2_1"

  ctx3._used_names["nonssa"] = True
  ctx3._used_names["_nonssa"] = True
  ctx3._used_names["v"] = True
  ctx3._used_names["_v"] = True
  ctx3._used_names["_v_0"] = True
  assert ctx3.register("nonssa") == "_v_1"


def test_generator_naming_from_type_attributes_and_collisions() -> None:
  """Verify `MlirToPythonGenerator` derives readable variable names from `type` attributes and resolves collisions."""
  op1 = OperationNode(
    name="sw.op",
    results=[ValueNode(name="%0")],
    attributes=[AttributeNode(name="type", value='"torch.flatten"')],
  )
  op2 = OperationNode(
    name="sw.op",
    results=[ValueNode(name="%1")],
    attributes=[AttributeNode(name="type", value='"torch.flatten"')],
  )
  op3 = OperationNode(
    name="sw.op",
    results=[ValueNode(name="%2")],
    attributes=[AttributeNode(name="type", value='"flax.nnx.Linear"')],
  )
  op4 = OperationNode(name="sw.op", results=[ValueNode(name="%a")])
  op_sink = OperationNode(
    name="sw.op",
    operands=[ValueNode(name="%0"), ValueNode(name="%1"), ValueNode(name="%2"), ValueNode(name="%a")],
    attributes=[AttributeNode(name="type", value='"nop"')],
  )
  code = _gen_code([op1, op2, op3, op4, op_sink])
  assert "_flatten = torch.flatten()" in code
  assert "_flatten_0 = torch.flatten()" in code
  assert "_linear = flax.nnx.Linear()" in code
  assert "_a =" in code
  assert "nop(_flatten, _flatten_0, _linear, _a)" in code
