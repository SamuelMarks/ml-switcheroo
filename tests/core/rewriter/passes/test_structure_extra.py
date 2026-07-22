"""Test suite for the Structure Extra module."""

import libcst as cst
from ml_switcheroo.core.rewriter.passes.structure import StructuralTransformer
from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo_ir.schema.ghost import SemanticTier
from ml_switcheroo.semantics.schema import StructuralTraits


class DummySemantics(SemanticsManager):
  """Dummy Semantics class for testing purposes."""

  def __init__(self):
    """Initializes the DummySemantics instance."""
    self.configs = {}
    self.framework_configs = self.configs
    self.definitions = {}
    self.variants = {}
    self.verified = True
    self.known_magic_args = set()

  def get_framework_config(self, fw):
    """Mock implementation of get framework configuration."""
    return self.configs.get(fw, {})

  def get_definition(self, name):
    """Mock implementation of get definition."""
    return self.definitions.get(name)

  def resolve_variant(self, abstract_id, fw):
    """Mock implementation of resolve variant."""
    return self.variants.get((abstract_id, fw))


def get_transformer():
  """Gets transformer."""
  semantics = DummySemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=True)
  ctx = RewriterContext(semantics, config)
  transformer = StructuralTransformer(ctx)
  return (transformer, semantics, ctx)


def test_target_traits_fallback():
  """Verifies the behavior of target traits fallback."""
  (transformer, sem, ctx) = get_transformer()
  traits = transformer.target_traits
  assert isinstance(traits, StructuralTraits)
  assert transformer._cached_target_traits is traits


def test_get_target_tiers_fallback():
  """Gets target tiers fallback."""
  (transformer, sem, ctx) = get_transformer()
  tiers = transformer._get_target_tiers()
  assert SemanticTier.ARRAY_API.value in tiers


def test_cst_to_string_fallback():
  """Verifies the behavior of cst to string fallback."""
  (transformer, sem, ctx) = get_transformer()
  node = cst.BinaryOperation(left=cst.Name("a"), operator=cst.Add(), right=cst.Name("b"))
  assert transformer._cst_to_string(node) is None
  assert transformer._get_qualified_name(node) is None


def test_is_framework_base_empty():
  """Checks if is framework base empty."""
  (transformer, sem, ctx) = get_transformer()
  assert transformer._is_framework_base("") is False
  assert transformer._is_framework_base(None) is False


def test_is_framework_base_traits_object():
  """Checks if is framework base traits object."""
  (transformer, sem, ctx) = get_transformer()

  class DummyTraits:
    """Dummy Traits class for testing purposes."""

    module_base = "my.Framework"

  sem.configs["torch"] = {"traits": DummyTraits()}
  assert transformer._is_framework_base("my.Framework") is True


def test_is_framework_base_suffix():
  """Checks if is framework base suffix."""
  (transformer, sem, ctx) = get_transformer()
  sem.configs["torch"] = {"traits": {"module_base": "torch.nn.Module"}}
  assert transformer._is_framework_base("nn.Module") is True
  assert transformer._is_framework_base("other.Module") is False


def test_get_source_inference_methods_fallback():
  """Gets source inference methods fallback."""
  (transformer, sem, ctx) = get_transformer()
  methods = transformer._get_source_inference_methods()
  assert "forward" in methods


def test_leave_module_preamble():
  """Verifies the behavior of leave module preamble."""
  (transformer, sem, ctx) = get_transformer()
  mod = cst.parse_module("a = 1")
  ctx.module_preamble.append("import sys")
  ctx.module_preamble.append("invalid code ###")
  new_mod = transformer.leave_Module(mod, mod)
  assert "import sys" in new_mod.code
  assert "invalid code" not in new_mod.code
  assert not ctx.module_preamble
  mod2 = transformer.leave_Module(new_mod, new_mod)
  assert mod2 is new_mod


def test_leave_name_not_in_annotation():
  """Verifies the behavior of leave name not in annotation."""
  (transformer, sem, ctx) = get_transformer()
  name = cst.Name("x")
  new_name = transformer.leave_Name(name, name)
  assert new_name is name


def test_leave_attribute_not_in_annotation():
  """Verifies the behavior of leave attribute not in annotation."""
  (transformer, sem, ctx) = get_transformer()
  attr = cst.Attribute(value=cst.Name("x"), attr=cst.Name("y"))
  new_attr = transformer.leave_Attribute(attr, attr)
  assert new_attr is attr


def test_visit_classdef_fallback_and_error():
  """Verifies the behavior of visit classdef fallback and correctly handling an error."""
  (transformer, sem, ctx) = get_transformer()
  sem.configs["torch"] = {"traits": {"module_base": "torch.nn.Module"}}
  class_node = cst.parse_module("class Net(nn.Module): pass").body[0]
  transformer.visit_ClassDef(class_node)
  assert ctx.in_module_class
  sem.configs["jax"] = {"tiers": ["array_api"]}
  ctx.current_stmt_errors.clear()
  transformer.visit_ClassDef(class_node)
  assert "does not support Neural Network classes" in ctx.current_stmt_errors[0]
  res = transformer.leave_ClassDef(class_node, class_node)
  assert isinstance(res, cst.FlattenSentinel)
  assert not ctx.in_module_class


def test_leave_classdef_unmapped_base():
  """Verifies the behavior of leave classdef unmapped base."""
  (transformer, sem, ctx) = get_transformer()
  sem.configs["torch"] = {"traits": {"module_base": "torch.nn.Module"}}
  sem.configs["jax"] = {"traits": {"module_base": "flax.nnx.Module"}}
  class_node = cst.parse_module("class Net(nn.Module, Other): pass").body[0]
  transformer.visit_ClassDef(class_node)
  new_node = transformer.leave_ClassDef(class_node, class_node)
  assert "flax.nnx.Module" in transformer._cst_to_string(new_node.bases[0].value)
  assert "Other" in transformer._cst_to_string(new_node.bases[1].value)


def test_leave_functiondef_no_stack():
  """Verifies the behavior of leave functiondef no stack."""
  (transformer, sem, ctx) = get_transformer()
  func = cst.parse_module("def foo(): pass").body[0]
  assert transformer.leave_FunctionDef(func, func) is func


def test_leave_functiondef_renaming():
  """Verifies the behavior of leave functiondef renaming."""
  (transformer, sem, ctx) = get_transformer()
  sem.configs["jax"] = {"traits": {"init_method_name": "setup"}}
  func = cst.parse_module("def __init__(self): pass").body[0]
  transformer.visit_FunctionDef(func)
  ctx.in_module_class = True
  ctx.signature_stack[-1].is_module_method = True
  new_func = transformer.leave_FunctionDef(func, func)
  assert new_func.name.value == "setup"


def test_leave_functiondef_magic_args():
  """Verifies the behavior of leave functiondef magic arguments."""
  (transformer, sem, ctx) = get_transformer()
  sem.known_magic_args.add("rngs")
  sem.configs["jax"] = {
    "traits": {"auto_strip_magic_args": True, "strip_magic_args": ["ctx"], "inject_magic_args": [("rngs", "int")]}
  }
  func = cst.parse_module("def __init__(self, ctx, rngs): pass").body[0]
  transformer.visit_FunctionDef(func)
  ctx.in_module_class = True
  ctx.signature_stack[-1].is_module_method = True
  new_func = transformer.leave_FunctionDef(func, func)
  params = [p.name.value for p in new_func.params.params if isinstance(p.name, cst.Name)]
  assert "ctx" not in params
  assert "rngs" in params


def test_super_init_logic():
  """Verifies the behavior of super initialization logic."""
  (transformer, sem, ctx) = get_transformer()
  sem.configs["jax"] = {"traits": {"requires_super_init": True}}
  func = cst.parse_module("def __init__(self): pass").body[0]
  transformer.visit_FunctionDef(func)
  ctx.in_module_class = True
  ctx.signature_stack[-1].is_module_method = True
  new_func = transformer.leave_FunctionDef(func, func)
  code = cst.Module([new_func]).code
  assert "super().__init__()" in code
  sem.configs["jax"] = {"traits": {"requires_super_init": False}}
  func2 = cst.parse_module("def __init__(self): pass").body[0]
  func2 = func2.with_changes(body=cst.SimpleStatementSuite(body=[cst.Pass()]))
  res = transformer._strip_super_init(func2)
  assert isinstance(res.body, cst.SimpleStatementSuite)
