"""Test suite for the Structure Extra2 module."""

import libcst as cst
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.core.rewriter.passes.structure import StructuralTransformer


class DummySemantics:
  """Dummy Semantics class for testing purposes."""

  def __init__(self):
    """Initializes the DummySemantics instance."""
    self.definitions = {}
    self.configs = {}
    self.framework_configs = self.configs
    self.variants = {}

  def resolve_definition(self, *args, **kwargs):
    """Mock implementation of resolve definition."""
    return None

  def get_standard_module(self, *args, **kwargs):
    """Mock implementation of get standard module."""
    return "nn"

  def resolve_variant(self, op_id, fw):
    """Mock implementation of resolve variant."""
    return self.variants.get((op_id, fw))

  def get_framework_config(self, framework):
    """Mock implementation of get framework configuration."""
    return self.configs.get(framework, {})

  def get_definition(self, api):
    """Mock implementation of get definition."""
    return self.definitions.get(api)


def get_transformer():
  """Gets transformer."""
  sem = DummySemantics()
  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  ctx = RewriterContext(sem, cfg)
  return (StructuralTransformer(ctx), sem, ctx)


def test_preamble_and_docstring():
  """Verifies the behavior of preamble and docstring."""
  (transformer, sem, ctx) = get_transformer()
  func = cst.parse_module('def __init__(self):\n  """doc"""\n  pass').body[0]
  transformer.visit_FunctionDef(func)
  ctx.in_module_class = True
  ctx.signature_stack[-1].is_module_method = True
  ctx.signature_stack[-1].injected_args.append(("y", "int"))
  ctx.signature_stack[-1].preamble_stmts.append("print(1)")
  ctx.signature_stack[-1].preamble_stmts.append("1 = 2")
  new_func = transformer.leave_FunctionDef(func, func)
  code = cst.Module([new_func]).code
  assert "print(1)" in code
  assert "y: Injected." in code
  assert "1 = 2" not in code
  func_simple = cst.parse_module("def foo(): print(2)").body[0]
  res_simple = transformer._convert_to_indented_block(func_simple)
  assert isinstance(res_simple.body, cst.IndentedBlock)


def test_strip_argument_from_signature():
  """Verifies the behavior of strip argument from signature."""
  (transformer, sem, ctx) = get_transformer()
  func = cst.parse_module("def f(x, y): pass").body[0]
  res = transformer._strip_argument_from_signature(func, "x")
  assert res.params.params[0].name.value == "y"


def test_fix_comma():
  """Fixes comma."""
  (transformer, sem, ctx) = get_transformer()
  func = cst.parse_module("def f(x, y): pass").body[0]
  params = list(func.params.params)
  params[-1] = params[-1].with_changes(comma=cst.Comma())
  res = transformer._fix_comma(func, params)
  assert res.params.params[-1].comma == cst.MaybeSentinel.DEFAULT


def test_leave_module_preamble_empty_stmts():
  """Verifies the behavior of leave module preamble empty stmts."""
  (transformer, sem, ctx) = get_transformer()
  mod = cst.parse_module("a = 1")
  ctx.module_preamble.append("invalid code ###")
  new_mod = transformer.leave_Module(mod, mod)
  assert "invalid code" not in new_mod.code
  assert new_mod is mod


def test_leave_name_in_annotation_success():
  """Verifies the behavior of leave name in annotation successfully."""
  (transformer, sem, ctx) = get_transformer()
  transformer._in_annotation = True
  sem.definitions["UnknownType"] = ("UnknownType", {})
  sem.variants["UnknownType", "jax"] = {"api": "jnp.MappedType"}
  name = cst.Name("UnknownType")
  res = transformer.leave_Name(name, name)
  assert transformer._cst_to_string(res) == "jnp.MappedType"


def test_leave_attribute_in_annotation_success():
  """Verifies the behavior of leave attribute in annotation successfully."""
  (transformer, sem, ctx) = get_transformer()
  transformer._in_annotation = True
  sem.definitions["Unknown.Type"] = ("UnknownType", {})
  sem.variants["UnknownType", "jax"] = {"api": "jnp.MappedType"}
  attr = cst.Attribute(value=cst.Name("Unknown"), attr=cst.Name("Type"))
  res = transformer.leave_Attribute(attr, attr)
  assert transformer._cst_to_string(res) == "jnp.MappedType"


def test_leave_classdef_raw_name_fallback():
  """Verifies the behavior of leave classdef raw name fallback."""
  (transformer, sem, ctx) = get_transformer()
  sem.configs["torch"] = {"traits": {"module_base": "torch.nn.Module"}}
  sem.configs["jax"] = {"traits": {"module_base": "flax.nnx.Module"}}
  class_node = cst.parse_module("class Net(nn.Module): pass").body[0]
  transformer.visit_ClassDef(class_node)
  orig_gqn = transformer._get_qualified_name
  transformer._get_qualified_name = lambda n: None
  new_node = transformer.leave_ClassDef(class_node, class_node)
  transformer._get_qualified_name = orig_gqn
  assert "flax.nnx.Module" in transformer._cst_to_string(new_node.bases[0].value)


def test_convert_to_indented_block_fallback():
  """Converts to indented block fallback."""
  (transformer, sem, ctx) = get_transformer()
  func = cst.parse_module("def foo():\n  pass").body[0]
  assert transformer._convert_to_indented_block(func) is func


def test_ensure_super_init_already_has():
  """Verifies the behavior of ensure super initialization already has."""
  (transformer, sem, ctx) = get_transformer()
  func = cst.parse_module("def __init__(self):\n  super().__init__()").body[0]
  assert transformer._ensure_super_init(func) is func


def test_strip_super_init_no_body():
  """Verifies the behavior of strip super initialization no body."""
  (transformer, sem, ctx) = get_transformer()
  func = cst.parse_module("def f(): pass").body[0]
  func = func.with_changes(body=cst.Pass())
  assert transformer._strip_super_init(func) is func


def test_has_super_init_false():
  """Checks if has super initialization false."""
  (transformer, sem, ctx) = get_transformer()
  func = cst.parse_module("def __init__(self):\n  pass").body[0]
  assert transformer._has_super_init(func) is False


def test_update_docstring_fallback():
  """Updates docstring fallback."""
  (transformer, sem, ctx) = get_transformer()
  func0 = cst.parse_module("def f(): pass").body[0]
  func0 = func0.with_changes(body=cst.SimpleStatementSuite(body=[]))
  assert transformer._update_docstring(func0, [("a", "b")]) is func0
  func = cst.parse_module("def f():\n  x = 1").body[0]
  assert transformer._update_docstring(func, [("a", "b")]) is func
  func2 = cst.parse_module("def f():\n  'doc'").body[0]
  assert transformer._update_docstring(func2, [("a", "b")]) is func2


def test_visit_classdef_raw_name_fallback():
  """Verifies the behavior of visit classdef raw name fallback."""
  (transformer, sem, ctx) = get_transformer()
  sem.configs["torch"] = {"traits": {"module_base": "torch.nn.Module"}}
  class_node = cst.parse_module("class Net(nn.Module): pass").body[0]
  orig_gqn = transformer._get_qualified_name
  transformer._get_qualified_name = lambda n: None
  transformer.visit_ClassDef(class_node)
  transformer._get_qualified_name = orig_gqn
  assert ctx.in_module_class


def test_leave_attribute_fallback_super():
  """Verifies the behavior of leave attribute fallback super."""
  (transformer, sem, ctx) = get_transformer()
  transformer._in_annotation = False
  attr = cst.Attribute(value=cst.Name("Unknown"), attr=cst.Name("Type"))
  res = transformer.leave_Attribute(attr, attr)
  assert res is attr
