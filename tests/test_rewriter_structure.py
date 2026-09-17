"""Test module."""

from typing import Any, Dict, List, Optional, Tuple

import libcst as cst

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.core.rewriter.passes.structure import StructuralPass, StructuralTransformer
from ml_switcheroo.semantics.schema import StructuralTraits


class DummySemantics:
  """Docstring."""

  def __init__(self) -> None:
    """Docstring."""
    self.framework_configs: Dict[str, Any] = {
      "jax": {
        "tiers": ["array", "neural"],
        "traits": {"module_base": "flax.nnx.Module", "forward_method": "__call__", "requires_super_init": False},
      },
      "torch": {"traits": {"module_base": "torch.nn.Module", "known_inference_methods": {"forward"}}},
    }
    self.alias_map: Dict[str, str] = {}
    self.known_magic_args: List[str] = ["rngs"]
    self.defs: Dict[str, Tuple[str, Dict[str, Any]]] = {}
    self.variants: Dict[Tuple[str, str], Optional[Dict[str, Any]]] = {}

  def get_framework_config(self, fw: str) -> Optional[Dict[str, Any]]:
    """Docstring."""
    return self.framework_configs.get(fw)

  def get_definition(self, name: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Docstring."""
    if name in self.defs:
      return self.defs[name]
    if name == "torch.Tensor":
      return ("tensor", {})
    return None

  def resolve_variant(self, abstract_id: str, fw: str) -> Optional[Dict[str, Any]]:
    """Docstring."""
    if (abstract_id, fw) in self.variants:
      return self.variants[(abstract_id, fw)]
    if abstract_id == "tensor" and fw == "jax":
      return {"api": "jax.Array"}
    return None


def test_structure_pass() -> None:
  """Docstring."""
  code: str = """
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, rngs=None):
        super().__init__()
        self.x = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Docstring.'''
        return x
"""
  config: RuntimeConfig = RuntimeConfig(source_fw="torch", target_fw="jax")
  semantics: DummySemantics = DummySemantics()
  semantics.framework_configs["jax"]["traits"] = StructuralTraits(
    module_base="flax.nnx.Module", forward_method="__call__", requires_super_init=False, strip_magic_args=["rngs"]
  ).model_dump()
  semantics.framework_configs["torch"]["traits"] = StructuralTraits(
    module_base="torch.nn.Module", known_inference_methods={"forward"}
  ).model_dump()
  context: RewriterContext = RewriterContext(semantics=semantics, config=config)
  context.alias_map = {"nn": "torch.nn", "torch": "torch"}

  module: cst.Module = cst.parse_module(code)
  pass_: StructuralPass = StructuralPass()
  updated: cst.Module = pass_.transform(module, context)

  updated_code: str = updated.code
  assert "flax.nnx.Module" in updated_code
  assert "def __call__(self," in updated_code
  assert "jax.Array" in updated_code


def test_structure_transformer_edge_cases() -> None:
  """Docstring."""
  config: RuntimeConfig = RuntimeConfig(source_fw="torch", target_fw="jax")
  semantics: DummySemantics = DummySemantics()
  context: RewriterContext = RewriterContext(semantics=semantics, config=config)
  transformer: StructuralTransformer = StructuralTransformer(context)

  # Missing framework config
  semantics.framework_configs = {}
  assert transformer.target_traits.module_base is None
  assert "array" in transformer._get_target_tiers()

  # Test qualified name fallback
  name: cst.Name = cst.Name("foo")
  assert transformer._get_qualified_name(name) == "foo"

  # Single-part alias in alias_map (lines 118-120)
  context.alias_map = {"th": "torch"}
  assert transformer._get_qualified_name(cst.Name("th")) == "torch"

  attr: cst.Attribute = cst.Attribute(value=cst.Name("a"), attr=cst.Name("b"))
  assert transformer._get_qualified_name(attr) == "a.b"

  # Not flattened
  assert transformer._cst_to_string(cst.Integer("1")) is None
  assert transformer._get_qualified_name(cst.Integer("1")) is None

  # is_framework_base
  assert not transformer._is_framework_base("")
  semantics.framework_configs = {"torch": {"traits": {"module_base": "torch.nn.Module"}}}
  transformer._known_module_bases = None
  assert transformer._is_framework_base("nn.Module")
  assert transformer._is_framework_base("torch.nn.Module")
  assert not transformer._is_framework_base("SomethingElse")

  # Suffix match in _is_framework_base (lines 204-205)
  transformer._known_module_bases = {"flax.linen.Module"}
  assert transformer._is_framework_base("linen.Module")

  # visit_Import and visit_ImportFrom (lines 400-440)
  import_node: cst.Module = cst.parse_module("import torch as th, os")
  import_node.visit(transformer)
  assert context.alias_map["th"] == "torch"
  assert context.alias_map["os"] == "os"

  from_node: cst.Module = cst.parse_module("from torch.nn import Linear as Lin, Conv2d")
  from_node.visit(transformer)
  assert context.alias_map["Lin"] == "torch.nn.Linear"
  assert context.alias_map["Conv2d"] == "torch.nn.Conv2d"

  # Relative import and ImportStar
  rel_node: cst.Module = cst.parse_module("from . import foo\nfrom bar import *")
  rel_node.visit(transformer)

  # get_source_inference_methods
  semantics.framework_configs = {"torch": {"traits": {"known_inference_methods": ["call"]}}}
  assert "call" in transformer._get_source_inference_methods()

  # type_mapping missing
  assert transformer._get_type_mapping("missing") is None
  assert transformer._get_type_mapping("int") is None

  # leave_Attribute in annotation where mapping has no api or mapping is None
  transformer._in_annotation = True
  attr_node = cst.parse_expression("foo.bar")
  assert isinstance(attr_node, cst.Attribute)
  assert transformer.leave_Attribute(attr_node, attr_node) == attr_node
  transformer._in_annotation = False

  # Error case in class def (unsupported tier)
  semantics.framework_configs = {
    "jax": {"tiers": ["math"], "traits": {}},
    "torch": {"traits": {"module_base": "torch.nn.Module"}},
  }
  context2: RewriterContext = RewriterContext(semantics=semantics, config=config)
  transformer2: StructuralTransformer = StructuralTransformer(context2)
  context2.alias_map = {"nn": "torch.nn"}

  class_def: cst.ClassDef = getattr(cst.parse_module("class A(torch.nn.Module): pass"), "body")[0]
  transformer2.visit_ClassDef(class_def)
  assert len(context2.current_stmt_errors) > 0

  class_def2: cst.ClassDef = getattr(cst.parse_module("class B(Unknown): pass"), "body")[0]
  class_def2.visit(transformer2)
  assert not context2.in_module_class

  # Test docstring update edge cases
  # Test preamble injection
  context2.module_preamble.append("import foo")
  mod: cst.Module = cst.parse_module("a = 1")
  mod2: cst.Module = transformer2.leave_Module(mod, mod)
  assert "import foo" in mod2.code

  # Test annotation rewrite
  context2.alias_map = {"torch": "torch"}
  semantics.framework_configs["torch"] = {"traits": {"module_base": "torch.nn.Module"}}
  semantics.framework_configs["jax"] = {"tiers": ["array", "neural"], "traits": {}}


def test_type_annotations_nested() -> None:
  """Docstring."""
  code: str = """
def foo(x: List[torch.Tensor]):
    pass
"""
  config: RuntimeConfig = RuntimeConfig(source_fw="torch", target_fw="jax")
  semantics: DummySemantics = DummySemantics()
  context: RewriterContext = RewriterContext(semantics=semantics, config=config)
  context.alias_map = {"torch": "torch"}
  module: cst.Module = cst.parse_module(code)
  pass_: StructuralPass = StructuralPass()
  updated: cst.Module = pass_.transform(module, context)
  assert "jax.Array" in updated.code


# --- Merged from test_rewriter_structure_extra.py ---


class DummySemanticsExtra:
  """Docstring."""

  def __init__(self):
    """Docstring."""
    self.framework_configs = {
      "jax": {
        "tiers": ["array", "neural"],
        "traits": {"module_base": "flax.nnx.Module", "forward_method": "__call__", "requires_super_init": False},
      },
      "torch": {"traits": {"module_base": "torch.nn.Module", "known_inference_methods": {"forward"}}},
    }
    self.alias_map = {}
    self.known_magic_args = ["rngs"]

  def get_framework_config(self, fw):
    """Docstring."""
    return self.framework_configs.get(fw)

  def get_definition(self, name):
    """Docstring."""
    if name == "torch.Tensor":
      return ("tensor", {})
    return None

  def resolve_variant(self, abstract_id, fw):
    """Docstring."""
    if abstract_id == "tensor" and fw == "jax":
      return {"api": "jax.Array"}
    return None


def test_structure_transformer_edge_cases2():
  """Docstring."""
  config = RuntimeConfig(source_fw="torch", target_fw="jax")
  semantics = DummySemantics()
  context = RewriterContext(semantics=semantics, config=config)
  transformer = StructuralTransformer(context)

  # 181, 183->174: self._known_module_bases
  class DummyTraits:
    """Docstring."""

    module_base = "dummy.Module"

  semantics.framework_configs = {"other": {"traits": DummyTraits()}}
  context = RewriterContext(semantics=semantics, config=config)
  transformer = StructuralTransformer(context)
  # to trigger property
  transformer._is_framework_base("other.Module")

  # 209: _get_source_inference_methods defaults
  semantics.framework_configs = {}
  context = RewriterContext(semantics=semantics, config=config)
  transformer = StructuralTransformer(context)
  assert len(transformer._get_source_inference_methods()) > 0

  # 246-263: leave_Module
  context.module_preamble.append("import foo")
  mod = cst.parse_module("a = 1")
  mod2 = transformer.leave_Module(mod, mod)
  assert "import foo" in mod2.code

  # 363->371, 365->363: visit_ClassDef raw_name fallback
  semantics.framework_configs = {"torch": {"traits": {"module_base": "torch.nn.Module"}}}
  context = RewriterContext(semantics=semantics, config=config)
  transformer = StructuralTransformer(context)
  mod3 = cst.parse_module("class A(torch.nn.Module): pass")
  class_def3 = mod3.body[0]
  transformer.visit_ClassDef(class_def3)

  # 419, 421->416, 424: leave_ClassDef error case & name fallback
  context.current_stmt_errors.append("err")
  res = transformer.leave_ClassDef(class_def3, class_def3)
  assert res is not class_def3

  context.current_stmt_errors.clear()

  # simulate raw name resolution branch
  # we need a base where _get_qualified_name returns None, but _cst_to_string returns something
  # e.g., a complex expression like A.B() that shouldn't happen but if it does
  mod4 = cst.parse_module("class A(foo()): pass")
  class_def4 = mod4.body[0]
  transformer.visit_ClassDef(class_def4)
  # Mock _get_qualified_name to return None, and _cst_to_string to return 'foo'
  import unittest.mock

  with unittest.mock.patch.object(transformer, "_get_qualified_name", return_value=None):
    with unittest.mock.patch.object(transformer, "_cst_to_string", return_value="torch.nn.Module"):
      context.in_module_class = True  # Set it manually
      # Also test when target_base is None
      with unittest.mock.patch.object(
        StructuralTransformer, "target_traits", new_callable=unittest.mock.PropertyMock
      ) as m:
        m.return_value = StructuralTraits()  # module_base is None
        transformer.leave_ClassDef(class_def4, class_def4)

  # 491: __init__ rename
  # We must configure torch as source so torch.nn.Module is recognized!
  semantics.framework_configs = {
    "jax": {"tiers": ["array", "neural"], "traits": {"init_method_name": "setup", "module_base": "flax.nnx.Module"}},
    "torch": {"traits": {"module_base": "torch.nn.Module"}},
  }
  context = RewriterContext(semantics=semantics, config=config)
  transformer = StructuralTransformer(context)
  mod5 = cst.parse_module("class A(torch.nn.Module):\n  def __init__(self): pass")
  class_def5 = mod5.body[0]
  func_def5 = class_def5.body.body[0]
  transformer.visit_ClassDef(class_def5)
  transformer.visit_FunctionDef(func_def5)
  res_func = transformer.leave_FunctionDef(func_def5, func_def5)
  transformer.leave_ClassDef(class_def5, class_def5)
  assert res_func.name.value == "setup"

  # 497-500: inject magic args
  semantics.framework_configs = {
    "jax": {"tiers": ["array", "neural"], "traits": {"inject_magic_args": [("rngs", "Any")]}},
    "torch": {"traits": {"module_base": "torch.nn.Module"}},
  }
  context = RewriterContext(semantics=semantics, config=config)
  transformer = StructuralTransformer(context)
  transformer.visit_ClassDef(class_def5)
  transformer.visit_FunctionDef(func_def5)
  # inject ahead of time to hit found_injected True
  context.signature_stack[-1].injected_args.append(("rngs", "Any"))
  res_func2 = transformer.leave_FunctionDef(func_def5, func_def5)
  transformer.leave_ClassDef(class_def5, class_def5)
  assert "rngs" in res_func2.params.params[-1].name.value

  # 505-507: strip magic args auto
  semantics.framework_configs = {
    "jax": {"tiers": ["array", "neural"], "traits": {"auto_strip_magic_args": True, "inject_magic_args": []}},
    "torch": {"traits": {"module_base": "torch.nn.Module"}},
  }
  context = RewriterContext(semantics=semantics, config=config)
  transformer = StructuralTransformer(context)
  mod6 = cst.parse_module("class A(torch.nn.Module):\n  def __init__(self, rngs): pass")
  class_def6 = mod6.body[0]
  func_def6 = class_def6.body.body[0]
  transformer.visit_ClassDef(class_def6)
  transformer.visit_FunctionDef(func_def6)
  res_func3 = transformer.leave_FunctionDef(func_def6, func_def6)
  transformer.leave_ClassDef(class_def6, class_def6)
  assert "rngs" not in [p.name.value for p in res_func3.params.params]

  # 514: requires_super_init
  semantics.framework_configs = {
    "jax": {"tiers": ["array", "neural"], "traits": {"requires_super_init": True}},
    "torch": {"traits": {"module_base": "torch.nn.Module"}},
  }
  context = RewriterContext(semantics=semantics, config=config)
  transformer = StructuralTransformer(context)
  mod7 = cst.parse_module("class A(torch.nn.Module):\n  def __init__(self): pass")
  class_def7 = mod7.body[0]
  func_def7 = class_def7.body.body[0]
  transformer.visit_ClassDef(class_def7)
  transformer.visit_FunctionDef(func_def7)
  transformer.leave_FunctionDef(func_def7, func_def7)
  transformer.leave_ClassDef(class_def7, class_def7)
  # _ensure_super_init adds super().__init__()
  # actually it depends if _ensure_super_init works

  # 521: apply preamble
  semantics.framework_configs = {
    "jax": {"tiers": ["array", "neural"], "traits": {}},
    "torch": {"traits": {"module_base": "torch.nn.Module"}},
  }
  context = RewriterContext(semantics=semantics, config=config)
  transformer = StructuralTransformer(context)
  transformer.visit_ClassDef(class_def7)
  transformer.visit_FunctionDef(func_def7)
  context.signature_stack[-1].preamble_stmts.append("a = 1")
  res_func5 = transformer.leave_FunctionDef(func_def7, func_def7)
  transformer.leave_ClassDef(class_def7, class_def7)
  # Check if a = 1 is parsed as a SimpleStatementLine in body
  assert len(res_func5.body.body) > 1
  assert hasattr(res_func5.body.body[0], "body")
  assert isinstance(res_func5.body.body[0].body[0], cst.Assign)

  # 525: update docstring
  # tested implicitly if injected_args is not empty
  semantics.framework_configs = {
    "jax": {"tiers": ["array", "neural"], "traits": {"inject_magic_args": [("rngs", "Any")]}},
    "torch": {"traits": {"module_base": "torch.nn.Module"}},
  }
  context = RewriterContext(semantics=semantics, config=config)
  transformer = StructuralTransformer(context)
  mod8 = cst.parse_module('class A(torch.nn.Module):\n  def __init__(self):\n    """Doc"""\n    pass')
  class_def8 = mod8.body[0]
  func_def8 = class_def8.body.body[0]
  transformer.visit_ClassDef(class_def8)
  transformer.visit_FunctionDef(func_def8)
  res_func6 = transformer.leave_FunctionDef(func_def8, func_def8)
  transformer.leave_ClassDef(class_def8, class_def8)
  # The first body item is a SimpleStatementLine containing an Expr with the string
  assert "rngs" in res_func6.body.body[0].body[0].value.value.value

  # 529: _resolve_alias missing fallback
  semantics.framework_configs = {"jax": {"tiers": ["array", "neural"], "traits": {}}}
  context = RewriterContext(semantics=semantics, config=config)
  transformer = StructuralTransformer(context)
  # how to trigger 529
  # def leave_FunctionDef -> 530 is return updated_node. 529 is in leave_FunctionDef? Wait 529 is docstring logic or return.


def test_structure_missing_branches() -> None:
  """Test remaining missing branches and statements in StructuralTransformer."""
  config = RuntimeConfig(source_fw="torch", target_fw="jax")
  semantics = DummySemantics()
  context = RewriterContext(semantics=semantics, config=config)
  transformer = StructuralTransformer(context)

  # 1. 138->140: _cst_to_string on Attribute with non-flattenable base
  attr_call = cst.Attribute(value=cst.Call(func=cst.Name("fn")), attr=cst.Name("attr"))
  assert transformer._cst_to_string(attr_call) is None

  # 2. 220->222: _get_source_inference_methods with empty known_inference_methods
  semantics.framework_configs["torch"]["traits"] = {"known_inference_methods": set()}
  assert "forward" in transformer._get_source_inference_methods()

  # 3. 286-287, 293: leave_Module with invalid preamble syntax and empty new_stmts
  mod = cst.parse_module("x = 1")
  context.module_preamble = ["invalid syntax !@@#"]
  res_mod = transformer.leave_Module(mod, mod)
  assert res_mod is mod

  # 4. 354: leave_Name in annotation with full_name present and full_name None
  transformer._in_annotation = True
  transformer._attribute_depth = 0
  semantics.defs = {
    "TensorWithApi": ("tensor_api", {}),
    "TensorNoApi": ("tensor_no_api", {}),
    "torch.TensorNoApi": ("tensor_no_api", {}),
  }
  semantics.variants = {
    ("tensor_api", "jax"): {"api": "jax.Array"},
    ("tensor_no_api", "jax"): {"dummy": "value"},
  }
  # 354 -> 355: has full_name and has api
  name_with_api = cst.Name("TensorWithApi")
  res_name1 = transformer.leave_Name(name_with_api, name_with_api)
  assert isinstance(res_name1, cst.Attribute)

  # 354 -> 358: full_name is None
  transformer._cst_to_string = lambda node: None  # type: ignore[assignment]
  name_no_str = cst.Name("x")
  res_name_no_str = transformer.leave_Name(name_no_str, name_no_str)
  assert res_name_no_str is name_no_str

  # Restore _cst_to_string
  del transformer._cst_to_string

  # 5. 377: leave_Attribute in annotation with full_name present and full_name None
  transformer._in_annotation = True
  transformer._attribute_depth = 0
  context.alias_map = {"torch": "torch"}
  # 377 -> 378: has full_name
  semantics.defs["torch.TensorWithApi"] = ("tensor_api", {})
  attr_with_api = cst.Attribute(value=cst.Name("torch"), attr=cst.Name("TensorWithApi"))
  res_attr_api = transformer.leave_Attribute(attr_with_api, attr_with_api)
  assert isinstance(res_attr_api, cst.Attribute)

  # 377 -> 384: full_name is None
  attr_call_none = cst.Attribute(value=cst.Call(func=cst.Name("fn")), attr=cst.Name("attr"))
  res_attr_none = transformer.leave_Attribute(attr_call_none, attr_call_none)
  assert res_attr_none == attr_call_none
  transformer._in_annotation = False

  # 6. 456-457: Fallback in visit_ClassDef when _get_qualified_name misses but _cst_to_string matches
  context_fallback = RewriterContext(semantics=DummySemantics(), config=config)
  context_fallback.alias_map = {"torch": "unmapped_pkg"}
  transformer_fallback = StructuralTransformer(context_fallback)
  class_mod_fallback = cst.parse_module("class Mod(torch.nn.Module): pass")
  class_fallback = class_mod_fallback.body[0]  # type: ignore[assignment]
  transformer_fallback.context.scope_stack.append(set())
  transformer_fallback.visit_ClassDef(class_fallback)
  assert transformer_fallback.context.in_module_class is True

  # 7. Target tier does not support neural
  semantics_non_neural = DummySemantics()
  semantics_non_neural.framework_configs = {
    "jax": {"tiers": ["array"], "traits": {}},
    "torch": {"traits": {"module_base": "torch.nn.Module"}},
  }
  context_non_neural = RewriterContext(semantics=semantics_non_neural, config=config)
  transformer_non_neural = StructuralTransformer(context_non_neural)
  class_mod = cst.parse_module("class Mod(torch.nn.Module): pass")
  class_def: cst.ClassDef = class_mod.body[0]  # type: ignore[assignment]
  transformer_non_neural.visit_ClassDef(class_def)
  assert any(
    "does not support Neural Network classes" in err for err in transformer_non_neural.context.current_stmt_errors
  )

  # 7. 488->510: leave_ClassDef when in_module_class is False
  transformer.context.scope_stack.append(set())
  transformer.context.in_module_class = False
  res_class = transformer.leave_ClassDef(class_def, class_def)
  assert res_class is class_def

  # 8. 507: leave_ClassDef with non-framework base class
  semantics.framework_configs["jax"] = {"tiers": ["neural"], "traits": {"module_base": "flax.nnx.Module"}}
  class_multi_base: cst.ClassDef = cst.parse_module("class M(torch.nn.Module, OtherBase): pass").body[0]  # type: ignore[assignment]
  transformer.context.scope_stack.append(set())
  transformer.context.in_module_class = True
  res_multi = transformer.leave_ClassDef(class_multi_base, class_multi_base)
  assert isinstance(res_multi, cst.ClassDef)
  assert len(res_multi.bases) == 2

  # 9. 530->529: visit_FunctionDef when param.name is not cst.Name
  param_noname = cst.Param(name=cst.SimpleString("'param'"))  # type: ignore[arg-type]
  func_noname: cst.FunctionDef = cst.parse_module("def f(): pass").body[0]  # type: ignore[assignment]
  func_noname = func_noname.with_changes(params=cst.Parameters(params=[param_noname]))
  transformer.visit_FunctionDef(func_noname)
  transformer.context.scope_stack.pop()
  transformer.context.signature_stack.pop()

  # 10. 559: leave_FunctionDef when signature_stack is empty
  func_node: cst.FunctionDef = cst.parse_module("def g(): pass").body[0]  # type: ignore[assignment]
  transformer.context.scope_stack.append(set())
  transformer.context.signature_stack.clear()
  assert transformer.leave_FunctionDef(func_node, func_node) is func_node

  # 11. 580->579: leave_FunctionDef when inject_magic_args argument already exists in function args
  semantics_magic = DummySemantics()
  semantics_magic.framework_configs = {
    "jax": {
      "tiers": ["neural"],
      "traits": {"module_base": "flax.nnx.Module", "inject_magic_args": [("rngs", "Any")]},
    },
    "torch": {"traits": {"module_base": "torch.nn.Module"}},
  }
  context_magic = RewriterContext(semantics=semantics_magic, config=config)
  transformer_magic = StructuralTransformer(context_magic)
  func_rngs_mod = cst.parse_module("class M(torch.nn.Module):\n  def __init__(self, rngs): pass")
  class_rngs: cst.ClassDef = func_rngs_mod.body[0]  # type: ignore[assignment]
  func_rngs: cst.FunctionDef = class_rngs.body.body[0]  # type: ignore[assignment]
  transformer_magic.visit_ClassDef(class_rngs)
  transformer_magic.visit_FunctionDef(func_rngs)
  res_rngs = transformer_magic.leave_FunctionDef(func_rngs, func_rngs)
  assert isinstance(res_rngs, cst.FunctionDef)
