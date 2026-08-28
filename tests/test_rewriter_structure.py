"""Test module."""

import libcst as cst
from ml_switcheroo.core.rewriter.passes.structure import StructuralPass, StructuralTransformer
from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.schema import StructuralTraits
from typing import Dict, List, Any, Optional, Tuple


class DummySemantics:
  """Test element."""

  def __init__(self) -> None:
    """Test element."""
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
    """Test element."""
    return self.framework_configs.get(fw)

  def get_definition(self, name: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Test element."""
    if name == "torch.Tensor":
      return ("tensor", {})
    return None

  def resolve_variant(self, abstract_id: str, fw: str) -> Optional[Dict[str, Any]]:
    """Test element."""
    if abstract_id == "tensor" and fw == "jax":
      return {"api": "jax.Array"}
    return None


def test_structure_pass() -> None:
  """Test element."""
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
  """Test element."""
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

  # get_source_inference_methods
  semantics.framework_configs = {"torch": {"traits": {"known_inference_methods": ["call"]}}}
  assert "call" in transformer._get_source_inference_methods()

  # type_mapping missing
  assert transformer._get_type_mapping("missing") is None

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
  """Test element."""
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
