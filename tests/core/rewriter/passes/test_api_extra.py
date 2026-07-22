"""Test suite for the Api Extra module."""

import libcst as cst
from ml_switcheroo.core.rewriter.passes.api import ApiTransformer
from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo_ir.schema.ghost import SemanticTier


class DummySemantics(SemanticsManager):
  """Dummy Semantics class for testing purposes."""

  framework_configs = {}

  def __init__(self):
    """Initializes the DummySemantics instance."""
    self.configs = {}
    self.definitions = {}
    self.variants = {}
    self.verified = True
    self._key_origins = {}

  def get_framework_config(self, fw):
    """Mock implementation of get framework configuration."""
    return self.configs.get(fw, {})

  def get_definition(self, name):
    """Mock implementation of get definition."""
    return self.definitions.get(name)

  def resolve_variant(self, abstract_id, fw):
    """Mock implementation of resolve variant."""
    return self.variants.get((abstract_id, fw))

  def is_verified(self, _id):
    """Mock implementation of is verified."""
    return self.verified


def get_transformer():
  """Gets transformer."""
  semantics = DummySemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=True)
  ctx = RewriterContext(semantics, config)
  ctx.hook_context = type(
    "MockHook", (), {"preamble_stmts_mock": [], "inject_preamble": lambda s, stmt: s.preamble_stmts_mock.append(stmt)}
  )()
  transformer = ApiTransformer(ctx)
  return (transformer, semantics, ctx)


def test_api_misc_helpers():
  """Verifies the behavior of API misc helpers."""
  (transformer, sem, ctx) = get_transformer()
  sem.configs["jax"] = {"traits": {"module_base": "jax.Module"}}
  traits = transformer._get_target_traits()
  assert traits is not None
  assert transformer._get_target_traits() is traits
  node = cst.BinaryOperation(left=cst.Name("a"), operator=cst.Add(), right=cst.Name("b"))
  assert transformer._cst_to_string(node) == "Add"
  ctx.alias_map["th"] = "torch"
  node_attr = cst.Attribute(value=cst.Name("th"), attr=cst.Name("nn"))
  assert transformer._get_qualified_name(node_attr) == "torch.nn"
  node_name = cst.Name("th")
  assert transformer._get_qualified_name(node_name) == "torch"
  node = transformer._create_dotted_name("a.b.c")
  assert isinstance(node, cst.Attribute)


def test_get_mapping():
  """Gets mapping."""
  (transformer, sem, ctx) = get_transformer()
  ctx.alias_map["torch"] = "torch"
  transformer._get_mapping("torch.missing", silent=False)
  assert "not found in semantics" in ctx.current_stmt_errors[0]
  ctx.current_stmt_errors.clear()
  sem.definitions["torch.unsafe"] = ("UnsafeOp", {})
  sem.verified = False
  transformer._get_mapping("torch.unsafe", silent=False)
  assert "Marked unsafe" in ctx.current_stmt_errors[0]
  ctx.current_stmt_errors.clear()
  sem.verified = True
  sem.definitions["torch.ok"] = ("OkOp", {})
  sem.variants["OkOp", "jax"] = {"api": "jnp.ok"}
  m = transformer._get_mapping("torch.ok", silent=False)
  assert m["api"] == "jnp.ok"
  sem.definitions["torch.nomap"] = ("NoMap", {})
  transformer._get_mapping("torch.nomap", silent=False)
  assert "No mapping available" in ctx.current_stmt_errors[0]
  ctx.current_stmt_errors.clear()


def test_handle_variant_imports():
  """Handles variant imports."""
  (transformer, sem, ctx) = get_transformer()
  var = {
    "required_imports": [
      "import os",
      "from sys import path",
      "json",
      {"module": "math"},
      {"module": "numpy", "alias": "np"},
    ]
  }
  transformer._handle_variant_imports(var)
  injected = ctx.hook_context.preamble_stmts_mock
  assert "import os" in injected
  assert "from sys import path" in injected
  assert "import json" in injected
  assert "import math" in injected
  assert "import numpy as np" in injected


def test_check_version_constraints():
  """Checks version constraints."""
  (transformer, sem, ctx) = get_transformer()
  assert transformer.check_version_constraints(None, None) is None
  sem.configs["jax"] = {"version": "0.4.1"}
  res = transformer.check_version_constraints(None, "0.3.0")
  assert "exceeds max" in res
  res = transformer.check_version_constraints("0.5.0", None)
  assert "older than" in res
  res = transformer.check_version_constraints("0.5.0", "0.6.0")
  assert "older than" in res
  assert transformer.check_version_constraints("0.4.0", "0.5.0") is None
  del sem.configs["jax"]
  transformer.context.config.target_framework = "flax_nnx"
  try:
    transformer.check_version_constraints("0.0.1", None)
  except Exception:
    pass


def test_is_framework_base():
  """Checks if is framework base."""
  (transformer, sem, ctx) = get_transformer()
  assert not transformer._is_framework_base("")
  sem.framework_configs["torch"] = {"traits": {"module_base": "torch.nn.Module"}}
  assert transformer._is_framework_base("torch.nn.Module")
  assert transformer._is_framework_base("nn.Module")
  assert not transformer._is_framework_base("Unknown")


def test_leave_module():
  """Verifies the behavior of leave module."""
  (transformer, sem, ctx) = get_transformer()
  ctx.module_preamble.append("import A")
  ctx.module_preamble.append("import A")
  mod = cst.parse_module("print(1)")
  new_mod = transformer.leave_Module(mod, mod)
  assert "import A" in new_mod.code


def test_stateful_scoping():
  """Verifies the behavior of stateful scoping."""
  (transformer, sem, ctx) = get_transformer()
  ctx.scope_stack.append(set())
  ctx.scope_stack.append(set())
  transformer._mark_stateful("my_var")
  assert "my_var" in ctx.scope_stack[-1]
  assert transformer._is_stateful("my_var")
  assert not transformer._is_stateful("other")


def test_visit_classdef_and_leave():
  """Verifies the behavior of visit classdef and leave."""
  (transformer, sem, ctx) = get_transformer()
  sem.framework_configs["torch"] = {"traits": {"module_base": "torch.nn.Module"}}
  class_node = cst.parse_module("class MyNet(torch.nn.Module):\n  pass").body[0]
  transformer.visit_ClassDef(class_node)
  assert ctx.in_module_class
  transformer.leave_ClassDef(class_node, class_node)
  assert not ctx.in_module_class


def test_visit_and_leave_functiondef():
  """Verifies the behavior of visit and leave functiondef."""
  (transformer, sem, ctx) = get_transformer()
  func_node = cst.parse_module("def __init__(self, x):\n  pass").body[0]
  transformer.visit_FunctionDef(func_node)
  assert len(ctx.signature_stack) == 1
  ctx.signature_stack[-1].injected_args.append(("y", "int"))
  ctx.signature_stack[-1].preamble_stmts.append("print('hi')")
  new_func = transformer.leave_FunctionDef(func_node, func_node)
  assert "y" in new_func.params.params[1].name.value
  assert "int" in new_func.params.params[1].annotation.annotation.value
  assert "print('hi')" in cst.Module([new_func]).code


def test_error_wrapping():
  """Verifies the behavior of correctly handling an error wrapping."""
  (transformer, sem, ctx) = get_transformer()
  stmt = cst.parse_module("x = 1").body[0]
  transformer.visit_SimpleStatementLine(stmt)
  ctx.current_stmt_errors.append("Error 1")
  ctx.current_stmt_errors.append("Error 1")
  new_stmt = transformer.leave_SimpleStatementLine(stmt, stmt)
  assert isinstance(new_stmt, cst.FlattenSentinel)
  transformer.visit_SimpleStatementLine(stmt)
  ctx.current_stmt_warnings.append("Warn 1")
  new_stmt2 = transformer.leave_SimpleStatementLine(stmt, stmt)
  assert isinstance(new_stmt2, cst.FlattenSentinel)


def test_imports():
  """Verifies the behavior of imports."""
  (transformer, sem, ctx) = get_transformer()
  imp = cst.parse_module("import a.b.c as d").body[0].body[0]
  transformer.visit_Import(imp)
  assert ctx.alias_map["d"] == "a.b.c"
  imp2 = cst.parse_module("import e.f").body[0].body[0]
  transformer.visit_Import(imp2)
  assert ctx.alias_map["e"] == "e"
  imp3 = cst.parse_module("from g.h import i as j, k").body[0].body[0]
  transformer.visit_ImportFrom(imp3)
  assert ctx.alias_map["j"] == "g.h.i"
  assert ctx.alias_map["k"] == "g.h.k"


def test_leave_assign_stateful():
  """Verifies the behavior of leave assign stateful."""
  (transformer, sem, ctx) = get_transformer()
  ctx.scope_stack.append(set())
  ctx.scope_stack.append(set())
  sem.definitions["torch.Linear"] = ("Linear", {})
  sem._key_origins["Linear"] = SemanticTier.NEURAL.value
  assign = cst.parse_module("self.layer = torch.Linear()").body[0].body[0]
  transformer.leave_Assign(assign, assign)
  assert "self.layer" in ctx.scope_stack[-2]
  assign2 = cst.parse_module("layer = torch.Linear()").body[0].body[0]
  transformer.leave_Assign(assign2, assign2)
  assert "layer" in ctx.scope_stack[-1]


def test_leave_assign_unwrapping():
  """Verifies the behavior of leave assign unwrapping."""
  (transformer, sem, ctx) = get_transformer()
  sem.framework_configs["torch"] = {"traits": {"functional_execution_method": "apply"}}
  assign3 = cst.parse_module("y, state = layer.apply()").body[0].body[0]
  new_assign3 = transformer.leave_Assign(assign3, assign3)
  assert isinstance(new_assign3.targets[0].target, cst.Name)
  assert new_assign3.targets[0].target.value == "y"


def test_leave_attribute():
  """Verifies the behavior of leave attribute."""
  (transformer, sem, ctx) = get_transformer()
  sem.definitions["torch.float32"] = ("float32", {"variants": {"jax": {"api": "jnp.float32"}}, "op_type": "constant"})
  sem.variants["float32", "jax"] = {"api": "jnp.float32"}
  attr = cst.parse_expression("torch.float32")
  new_attr = transformer.leave_Attribute(attr, attr)
  assert new_attr.attr.value == "float32"
  sem.definitions["torch.func_attr"] = ("func_attr", {"op_type": "function", "std_args": ["x"]})
  attr2 = cst.parse_expression("torch.func_attr")
  new_attr2 = transformer.leave_Attribute(attr2, attr2)
  assert new_attr2 is attr2
  sem.definitions["torch.inf"] = ("inf", {"op_type": "constant"})
  sem.variants["inf", "jax"] = {"macro_template": "float('inf')"}
  attr3 = cst.parse_expression("torch.inf")
  new_attr3 = transformer.leave_Attribute(attr3, attr3)
  assert new_attr3.args[0].value.value == "'inf'"


def test_leave_call_fallback_and_warnings():
  """Verifies the behavior of leave call fallback and warnings."""
  (transformer, sem, ctx) = get_transformer()
  sem.definitions["torch.deprecated"] = ("DepOp", {"deprecated": True, "replaced_by": "torch.new_op"})
  sem.variants["DepOp", "jax"] = {"api": "jnp.new_op"}
  call = cst.parse_expression("torch.deprecated()")
  transformer.leave_Call(call, call)
  assert "Consider using 'torch.new_op' instead" in ctx.current_stmt_warnings[0]


def test_normalize_arguments_pack_variadics():
  """Verifies the behavior of normalize arguments pack variadics."""
  (transformer, sem, ctx) = get_transformer()
  op_details = {"std_args": ["x", {"name": "dim", "is_variadic": True}], "variants": {"torch": {"args": {"dim": "dim"}}}}
  target_impl = {"api": "jnp.sum", "pack_to_tuple": "axis", "pack_as": "Tuple", "args": {"dim": "axis"}}
  call = cst.parse_expression("torch.sum(x, 1, 2)")
  args = transformer._normalize_arguments(call, call, op_details, target_impl)
  assert len(args) == 2
  assert args[1].keyword.value == "axis"
  assert isinstance(args[1].value, cst.Tuple)
  target_impl["pack_as"] = "List"
  args = transformer._normalize_arguments(call, call, op_details, target_impl)
  assert isinstance(args[1].value, cst.List)


def test_normalize_arguments_inject_and_kwargs_map():
  """Verifies the behavior of normalize arguments inject and keyword arguments map."""
  (transformer, sem, ctx) = get_transformer()
  op_details = {"std_args": ["x"], "variants": {"torch": {"args": {}}}}
  target_impl = {
    "api": "jnp.foo",
    "kwargs_map": {"drop_me": None},
    "inject_args": {"injected": "True"},
    "arg_values": {"injected2": {"1": "True"}},
  }
  call = cst.parse_expression("torch.foo(x, drop_me=1)")
  args = transformer._normalize_arguments(call, call, op_details, target_impl)
  keys = [a.keyword.value for a in args if a.keyword]
  assert "drop_me" not in keys
  assert "injected" in keys
  call2 = cst.parse_expression("x.add(y)")
  transformer._normalize_arguments(call2, call2, op_details, target_impl)


def test_api_convert_indented_block_no_op():
  """Verifies the behavior of API convert indented block no op."""
  from ml_switcheroo.core.rewriter.passes.api import ApiTransformer
  from ml_switcheroo.core.rewriter.context import RewriterContext
  import libcst as cst
  from unittest.mock import MagicMock
  from ml_switcheroo.config import RuntimeConfig
  from ml_switcheroo.semantics.manager import SemanticsManager

  ctx = RewriterContext(SemanticsManager(), RuntimeConfig(), MagicMock())
  t = ApiTransformer(ctx)
  node = cst.parse_module("def foo():\n  pass\n").body[0]
  res = t._convert_to_indented_block(node)
  assert res is node
