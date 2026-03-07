"""Module docstring."""

import pytest
import libcst as cst
from typing import Dict, Any

from ml_switcheroo.core.rewriter.passes.structure import StructuralPass, StructuralTransformer
from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.enums import SemanticTier
from ml_switcheroo.core.rewriter.types import SignatureContext
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


def test_preamble_and_docstring():
  """Function docstring."""
  transformer, sem, ctx = get_transformer()
  # Lines 387, 434-442, 446-458, 462-465, 510, 518-528
  func = cst.parse_module('def __init__(self):\n  """doc"""\n  pass').body[0]
  transformer.visit_FunctionDef(func)
  ctx.in_module_class = True
  ctx.signature_stack[-1].is_module_method = True
  ctx.signature_stack[-1].injected_args.append(("y", "int"))
  ctx.signature_stack[-1].preamble_stmts.append("print(1)")
  ctx.signature_stack[-1].preamble_stmts.append("1 = 2")  # exception line 434

  new_func = transformer.leave_FunctionDef(func, func)
  code = cst.Module([new_func]).code
  assert "print(1)" in code
  assert "y: Injected." in code
  assert "1 = 2" not in code

  # Line 462-465: simple suite
  func_simple = cst.parse_module("def foo(): print(2)").body[0]
  res_simple = transformer._convert_to_indented_block(func_simple)
  assert isinstance(res_simple.body, cst.IndentedBlock)


def test_strip_argument_from_signature():
  """Function docstring."""
  transformer, sem, ctx = get_transformer()
  # Lines 399-401
  func = cst.parse_module("def f(x, y): pass").body[0]
  res = transformer._strip_argument_from_signature(func, "x")
  assert res.params.params[0].name.value == "y"


def test_fix_comma():
  """Function docstring."""
  transformer, sem, ctx = get_transformer()
  # Line 427
  func = cst.parse_module("def f(x, y): pass").body[0]
  params = list(func.params.params)
  params[-1] = params[-1].with_changes(comma=cst.Comma())
  res = transformer._fix_comma(func, params)
  assert res.params.params[-1].comma == cst.MaybeSentinel.DEFAULT


def test_leave_module_preamble_empty_stmts():
  """Function docstring."""
  transformer, sem, ctx = get_transformer()
  mod = cst.parse_module("a = 1")
  ctx.module_preamble.append("invalid code ###")
  new_mod = transformer.leave_Module(mod, mod)
  assert "invalid code" not in new_mod.code
  assert new_mod is mod


def test_leave_name_in_annotation_success():
  """Function docstring."""
  transformer, sem, ctx = get_transformer()
  # Line 221
  transformer._in_annotation = True
  sem.definitions["UnknownType"] = ("UnknownType", {})
  sem.variants[("UnknownType", "jax")] = {"api": "jnp.MappedType"}
  name = cst.Name("UnknownType")
  res = transformer.leave_Name(name, name)
  assert transformer._cst_to_string(res) == "jnp.MappedType"


def test_leave_attribute_in_annotation_success():
  """Function docstring."""
  transformer, sem, ctx = get_transformer()
  # Line 245
  transformer._in_annotation = True
  sem.definitions["Unknown.Type"] = ("UnknownType", {})
  sem.variants[("UnknownType", "jax")] = {"api": "jnp.MappedType"}
  attr = cst.Attribute(value=cst.Name("Unknown"), attr=cst.Name("Type"))
  res = transformer.leave_Attribute(attr, attr)
  assert transformer._cst_to_string(res) == "jnp.MappedType"


def test_leave_classdef_raw_name_fallback():
  """Function docstring."""
  transformer, sem, ctx = get_transformer()
  # Line 301
  sem.configs["torch"] = {"traits": {"module_base": "torch.nn.Module"}}
  sem.configs["jax"] = {"traits": {"module_base": "flax.nnx.Module"}}
  class_node = cst.parse_module("class Net(nn.Module): pass").body[0]
  transformer.visit_ClassDef(class_node)

  # Mock _get_qualified_name to return None
  orig_gqn = transformer._get_qualified_name
  transformer._get_qualified_name = lambda n: None
  new_node = transformer.leave_ClassDef(class_node, class_node)
  transformer._get_qualified_name = orig_gqn

  assert "flax.nnx.Module" in transformer._cst_to_string(new_node.bases[0].value)


def test_convert_to_indented_block_fallback():
  """Function docstring."""
  transformer, sem, ctx = get_transformer()
  # Line 465
  func = cst.parse_module("def foo():\n  pass").body[0]
  assert transformer._convert_to_indented_block(func) is func


def test_ensure_super_init_already_has():
  """Function docstring."""
  transformer, sem, ctx = get_transformer()
  # Line 470
  func = cst.parse_module("def __init__(self):\n  super().__init__()").body[0]
  assert transformer._ensure_super_init(func) is func


def test_strip_super_init_no_body():
  """Function docstring."""
  transformer, sem, ctx = get_transformer()
  # Line 483
  func = cst.parse_module("def f(): pass").body[0]
  func = func.with_changes(body=cst.Pass())  # force no body attr
  assert transformer._strip_super_init(func) is func


def test_has_super_init_false():
  """Function docstring."""
  transformer, sem, ctx = get_transformer()
  # Line 493
  func = cst.parse_module("def __init__(self):\n  pass").body[0]
  assert transformer._has_super_init(func) is False


def test_update_docstring_fallback():
  """Function docstring."""
  transformer, sem, ctx = get_transformer()
  # Line 510: empty body
  func0 = cst.parse_module("def f(): pass").body[0]
  func0 = func0.with_changes(body=cst.SimpleStatementSuite(body=[]))
  assert transformer._update_docstring(func0, [("a", "b")]) is func0

  # not simple string
  func = cst.parse_module("def f():\n  x = 1").body[0]
  assert transformer._update_docstring(func, [("a", "b")]) is func

  # Line 528: no \"\"\" in docstring
  func2 = cst.parse_module("def f():\n  'doc'").body[0]
  assert transformer._update_docstring(func2, [("a", "b")]) is func2


def test_visit_classdef_raw_name_fallback():
  """Function docstring."""
  transformer, sem, ctx = get_transformer()
  # Lines 264-267
  sem.configs["torch"] = {"traits": {"module_base": "torch.nn.Module"}}
  class_node = cst.parse_module("class Net(nn.Module): pass").body[0]

  # Mock _get_qualified_name to return None to hit raw_name fallback
  orig_gqn = transformer._get_qualified_name
  transformer._get_qualified_name = lambda n: None
  transformer.visit_ClassDef(class_node)
  transformer._get_qualified_name = orig_gqn

  assert ctx.in_module_class


def test_leave_attribute_fallback_super():
  """Function docstring."""
  transformer, sem, ctx = get_transformer()
  # Line 245: fallthrough to super()
  transformer._in_annotation = False  # or True with no mapping
  attr = cst.Attribute(value=cst.Name("Unknown"), attr=cst.Name("Type"))
  res = transformer.leave_Attribute(attr, attr)
  assert res is attr
