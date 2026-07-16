"""Module docstring."""

import libcst as cst

from ml_switcheroo.core.rewriter.passes.structure import StructuralTransformer
from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo_ir.schema.ghost import SemanticTier
from ml_switcheroo.semantics.schema import StructuralTraits


class DummySemantics(SemanticsManager):
  """Class docstring."""

  def __init__(self):
    """Function docstring."""
    self.configs = {}
    self.framework_configs = self.configs
    self.definitions = {}
    self.variants = {}
    self.verified = True
    self.known_magic_args = set()

  def get_framework_config(self, fw):
    """Function docstring."""
    return self.configs.get(fw, {})

  def get_definition(self, name):
    """Function docstring."""
    return self.definitions.get(name)

  def resolve_variant(self, abstract_id, fw):
    """Function docstring."""
    return self.variants.get((abstract_id, fw))


def get_transformer():
  """Function docstring."""
  semantics = DummySemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=True)
  ctx = RewriterContext(semantics, config)
  transformer = StructuralTransformer(ctx)
  return transformer, semantics, ctx


def test_target_traits_fallback():
  """Function docstring."""
  transformer, sem, ctx = get_transformer()
  # Line 79
  traits = transformer.target_traits
  assert isinstance(traits, StructuralTraits)
  assert transformer._cached_target_traits is traits


def test_get_target_tiers_fallback():
  """Function docstring."""
  transformer, sem, ctx = get_transformer()
  # Line 86
  tiers = transformer._get_target_tiers()
  assert SemanticTier.ARRAY_API.value in tiers


def test_cst_to_string_fallback():
  """Function docstring."""
  transformer, sem, ctx = get_transformer()
  # Line 114
  node = cst.BinaryOperation(left=cst.Name("a"), operator=cst.Add(), right=cst.Name("b"))
  assert transformer._cst_to_string(node) is None
  # Line 93: get_qualified_name returns None when _cst_to_string is falsey
  assert transformer._get_qualified_name(node) is None


def test_is_framework_base_empty():
  """Function docstring."""
  transformer, sem, ctx = get_transformer()
  # Line 127
  assert transformer._is_framework_base("") is False
  assert transformer._is_framework_base(None) is False


def test_is_framework_base_traits_object():
  """Function docstring."""
  transformer, sem, ctx = get_transformer()

  # Line 139: getattr(traits, "module_base")
  class DummyTraits:
    """Class docstring."""

    module_base = "my.Framework"

  sem.configs["torch"] = {"traits": DummyTraits()}
  assert transformer._is_framework_base("my.Framework") is True


def test_is_framework_base_suffix():
  """Function docstring."""
  transformer, sem, ctx = get_transformer()
  # Lines 148-152: suffix check
  sem.configs["torch"] = {"traits": {"module_base": "torch.nn.Module"}}
  assert transformer._is_framework_base("nn.Module") is True
  assert transformer._is_framework_base("other.Module") is False


def test_get_source_inference_methods_fallback():
  """Function docstring."""
  transformer, sem, ctx = get_transformer()
  # Line 162
  methods = transformer._get_source_inference_methods()
  assert "forward" in methods


def test_leave_module_preamble():
  """Function docstring."""
  transformer, sem, ctx = get_transformer()
  # Lines 183-200
  mod = cst.parse_module("a = 1")
  ctx.module_preamble.append("import sys")
  ctx.module_preamble.append("invalid code ###")  # exception path line 183
  new_mod = transformer.leave_Module(mod, mod)
  assert "import sys" in new_mod.code
  assert "invalid code" not in new_mod.code
  # line 195: clear and return updated_node if empty
  assert not ctx.module_preamble
  mod2 = transformer.leave_Module(new_mod, new_mod)
  assert mod2 is new_mod


def test_leave_name_not_in_annotation():
  """Function docstring."""
  transformer, sem, ctx = get_transformer()
  # Line 221
  name = cst.Name("x")
  new_name = transformer.leave_Name(name, name)
  assert new_name is name


def test_leave_attribute_not_in_annotation():
  """Function docstring."""
  transformer, sem, ctx = get_transformer()
  # Line 245
  attr = cst.Attribute(value=cst.Name("x"), attr=cst.Name("y"))
  new_attr = transformer.leave_Attribute(attr, attr)
  assert new_attr is attr


def test_visit_classdef_fallback_and_error():
  """Function docstring."""
  transformer, sem, ctx = get_transformer()
  # Lines 264-267: raw_name fallback
  sem.configs["torch"] = {"traits": {"module_base": "torch.nn.Module"}}
  class_node = cst.parse_module("class Net(nn.Module): pass").body[0]
  transformer.visit_ClassDef(class_node)
  assert ctx.in_module_class

  # Line 274: error for unsupported tier
  sem.configs["jax"] = {"tiers": ["array_api"]}
  ctx.current_stmt_errors.clear()
  transformer.visit_ClassDef(class_node)
  assert "does not support Neural Network classes" in ctx.current_stmt_errors[0]

  # Lines 289-291: leave_ClassDef error handling
  res = transformer.leave_ClassDef(class_node, class_node)
  assert isinstance(res, cst.FlattenSentinel)  # EscapeHatch
  assert not ctx.in_module_class


def test_leave_classdef_unmapped_base():
  """Function docstring."""
  transformer, sem, ctx = get_transformer()
  sem.configs["torch"] = {"traits": {"module_base": "torch.nn.Module"}}
  sem.configs["jax"] = {"traits": {"module_base": "flax.nnx.Module"}}
  # Lines 301, 308: name fallback, other base kept
  class_node = cst.parse_module("class Net(nn.Module, Other): pass").body[0]
  transformer.visit_ClassDef(class_node)
  new_node = transformer.leave_ClassDef(class_node, class_node)
  assert "flax.nnx.Module" in transformer._cst_to_string(new_node.bases[0].value)
  assert "Other" in transformer._cst_to_string(new_node.bases[1].value)


def test_leave_functiondef_no_stack():
  """Function docstring."""
  transformer, sem, ctx = get_transformer()
  # Line 338
  func = cst.parse_module("def foo(): pass").body[0]
  assert transformer.leave_FunctionDef(func, func) is func


def test_leave_functiondef_renaming():
  """Function docstring."""
  transformer, sem, ctx = get_transformer()
  # Line 353: init renaming
  sem.configs["jax"] = {"traits": {"init_method_name": "setup"}}
  func = cst.parse_module("def __init__(self): pass").body[0]
  transformer.visit_FunctionDef(func)
  ctx.in_module_class = True
  ctx.signature_stack[-1].is_module_method = True
  new_func = transformer.leave_FunctionDef(func, func)
  assert new_func.name.value == "setup"


def test_leave_functiondef_magic_args():
  """Function docstring."""
  transformer, sem, ctx = get_transformer()
  # Lines 367-369: auto strip
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
  assert "rngs" in params  # injected magic arg should not be stripped!


def test_super_init_logic():
  """Function docstring."""
  transformer, sem, ctx = get_transformer()
  # Lines 376, 469-476, 490-494
  sem.configs["jax"] = {"traits": {"requires_super_init": True}}
  func = cst.parse_module("def __init__(self): pass").body[0]
  transformer.visit_FunctionDef(func)
  ctx.in_module_class = True
  ctx.signature_stack[-1].is_module_method = True
  new_func = transformer.leave_FunctionDef(func, func)
  code = cst.Module([new_func]).code
  assert "super().__init__()" in code

  # Line 483: strip super init from empty body
  sem.configs["jax"] = {"traits": {"requires_super_init": False}}
  func2 = cst.parse_module("def __init__(self): pass").body[0]
  # make body lack .body
  func2 = func2.with_changes(body=cst.SimpleStatementSuite(body=[cst.Pass()]))
  res = transformer._strip_super_init(func2)
  assert isinstance(res.body, cst.SimpleStatementSuite)
