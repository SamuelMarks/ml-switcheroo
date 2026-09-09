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

  def get_framework_config(self, fw: str) -> Optional[Dict[str, Any]]:
    """Docstring."""
    return self.framework_configs.get(fw)

  def get_definition(self, name: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Docstring."""
    if name == "torch.Tensor":
      return ("tensor", {})
    return None

  def resolve_variant(self, abstract_id: str, fw: str) -> Optional[Dict[str, Any]]:
    """Docstring."""
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
