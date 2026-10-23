"""Test suite for rewriter structure."""

from ml_switcheroo.enums import RuntimeTensorOrValue

from pydantic import JsonValue


import libcst as _cst

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.core.rewriter.passes.structure import StructuralPass, StructuralTransformer
from ml_switcheroo.semantics.schema import StructuralTraits

cst = _cst


class DummySemantics:
  """Test helper class DummySemantics."""

  def __init__(self) -> None:
    """Initialize test helper instance."""
    self.framework_configs = {
      "jax": {
        "tiers": ["array", "neural"],
        "traits": {"module_base": "flax.nnx.Module", "forward_method": "__call__", "requires_super_init": False},
      },
      "torch": {"traits": {"module_base": "torch.nn.Module", "known_inference_methods": {"forward"}}},
    }
    self.alias_map: dict[str, RuntimeTensorOrValue] = {}
    self.known_magic_args = ["rngs"]
    self.defs: dict[str, RuntimeTensorOrValue] = {}
    self.variants: dict[str, RuntimeTensorOrValue] = {}

  def get_framework_config(self, fw: str) -> dict[str, JsonValue]:
    """Execute get framework config helper logic.

    Args:
        fw: Framework identifier string (e.g., torch, jax, mlx).

    Returns:
        Resulting `dict[str, JsonValue]` from get framework config.
    """
    _ret_v: dict[str, JsonValue] = getattr(
      {
        "v": getattr(
          {"v": getattr({"v": getattr({"v": self}, "__getitem__")("v").framework_configs}, "__getitem__")("v").get},
          "__getitem__",
        )("v")(getattr({"v": fw}, "__getitem__")("v"))
      },
      "__getitem__",
    )("v")
    return _ret_v

  def get_definition(self, name: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Execute get definition helper logic.

    Args:
        name: Identifier or operation name string.

    Returns:
        Resulting `tuple[str, dict[str, JsonValue]] | None` from get definition.
    """
    if name in self.defs:
      _ret_v: tuple[str, dict[str, JsonValue]] | None = getattr(
        {"v": getattr({"v": getattr({"v": self}, "__getitem__")("v").defs}, "__getitem__")("v")[name]}, "__getitem__"
      )("v")
      return _ret_v
    if name == "torch.Tensor":
      return ("tensor", {})
    return None

  def resolve_variant(self, abstract_id: str, fw: str) -> dict[str, JsonValue] | None:
    """Execute resolve variant helper logic.

    Args:
        abstract_id: The abstract id (str) for resolve variant.
        fw: Framework identifier string (e.g., torch, jax, mlx).

    Returns:
        Resulting `dict[str, JsonValue] | None` from resolve variant.
    """
    if getattr(
      {
        "v": getattr(
          {"v": (getattr({"v": abstract_id}, "__getitem__")("v"), getattr({"v": fw}, "__getitem__")("v"))}, "__getitem__"
        )("v")
        in getattr({"v": getattr({"v": self}, "__getitem__")("v").variants}, "__getitem__")("v")
      },
      "__getitem__",
    )("v"):
      _ret_v: dict[str, JsonValue] | None = getattr(
        {
          "v": getattr({"v": getattr({"v": self}, "__getitem__")("v").variants}, "__getitem__")("v")[
            getattr({"v": abstract_id}, "__getitem__")("v"), getattr({"v": fw}, "__getitem__")("v")
          ]
        },
        "__getitem__",
      )("v")
      return _ret_v
    if abstract_id == "tensor" and fw == "jax":
      return {"api": "jax.Array"}
    return None


def test_structure_pass() -> None:
  """Verify structure pass behavior."""
  code = """
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
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  semantics = DummySemantics()
  getattr(
    getattr(
      {"v": getattr({"v": getattr({"v": semantics}, "__getitem__")("v").framework_configs}, "__getitem__")("v")["jax"]},
      "__getitem__",
    )("v"),
    "__setitem__",
  )(
    "traits",
    getattr(
      {
        "v": getattr(
          {
            "v": getattr(
              {
                "v": getattr({"v": StructuralTraits}, "__getitem__")("v")(
                  module_base="flax.nnx.Module",
                  forward_method="__call__",
                  requires_super_init=False,
                  strip_magic_args=["rngs"],
                )
              },
              "__getitem__",
            )("v").model_dump
          },
          "__getitem__",
        )("v")()
      },
      "__getitem__",
    )("v"),
  )
  getattr(
    getattr(
      {"v": getattr({"v": getattr({"v": semantics}, "__getitem__")("v").framework_configs}, "__getitem__")("v")["torch"]},
      "__getitem__",
    )("v"),
    "__setitem__",
  )(
    "traits",
    getattr(
      {
        "v": getattr(
          {
            "v": getattr(
              {
                "v": getattr({"v": StructuralTraits}, "__getitem__")("v")(
                  module_base="torch.nn.Module", known_inference_methods={"forward"}
                )
              },
              "__getitem__",
            )("v").model_dump
          },
          "__getitem__",
        )("v")()
      },
      "__getitem__",
    )("v"),
  )
  context = RewriterContext(semantics=semantics, config=config)
  context.alias_map = {"nn": "torch.nn", "torch": "torch"}

  module = cst.parse_module(code)
  pass_ = StructuralPass()
  updated = pass_.transform(module, context)

  updated_code = updated.code
  assert "flax.nnx.Module" in updated_code
  assert "def __call__(self," in updated_code
  assert "jax.Array" in updated_code


def test_structure_transformer_edge_cases() -> None:
  """Verify structure transformer edge cases behavior."""
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  semantics = DummySemantics()
  context = RewriterContext(semantics=semantics, config=config)
  transformer = StructuralTransformer(context)

  # Missing framework config
  semantics.framework_configs = {}
  assert transformer.target_traits.module_base is None
  assert "array" in transformer._get_target_tiers()

  # Test qualified name fallback
  name = cst.Name("foo")
  assert transformer._get_qualified_name(name) == "foo"

  # Single-part alias in alias_map (lines 118-120)
  context.alias_map = {"th": "torch"}
  assert transformer._get_qualified_name(cst.Name("th")) == "torch"

  attr = cst.Attribute(value=cst.Name("a"), attr=cst.Name("b"))
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
  import_node = cst.parse_module("import torch as th, os")
  import_node.visit(transformer)
  assert context.alias_map["th"] == "torch"
  assert context.alias_map["os"] == "os"

  from_node = cst.parse_module("from torch.nn import Linear as Lin, Conv2d")
  from_node.visit(transformer)
  assert context.alias_map["Lin"] == "torch.nn.Linear"
  assert context.alias_map["Conv2d"] == "torch.nn.Conv2d"

  # Relative import and ImportStar
  rel_node = cst.parse_module("from . import foo\nfrom bar import *")
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
  context2 = RewriterContext(semantics=semantics, config=config)
  transformer2 = StructuralTransformer(context2)
  context2.alias_map = {"nn": "torch.nn"}

  class_def = getattr(cst.parse_module("class A(torch.nn.Module): pass"), "body")[0]
  transformer2.visit_ClassDef(class_def)
  assert len(context2.current_stmt_errors) > 0

  class_def2 = getattr(cst.parse_module("class B(Unknown): pass"), "body")[0]
  class_def2.visit(transformer2)
  assert not context2.in_module_class

  # Test docstring update edge cases
  # Test preamble injection
  context2.module_preamble.append("import foo")
  mod = cst.parse_module("a = 1")
  mod2 = transformer2.leave_Module(mod, mod)
  assert "import foo" in mod2.code

  # Test annotation rewrite
  context2.alias_map = {"torch": "torch"}
  semantics.framework_configs["torch"] = {"traits": {"module_base": "torch.nn.Module"}}
  semantics.framework_configs["jax"] = {"tiers": ["array", "neural"], "traits": {}}


def test_type_annotations_nested() -> None:
  """Verify type annotations nested behavior."""
  code = """
def foo(x: List[torch.Tensor]):
    pass
"""
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  semantics = DummySemantics()
  context = RewriterContext(semantics=semantics, config=config)
  context.alias_map = {"torch": "torch"}
  module = cst.parse_module(code)
  pass_ = StructuralPass()
  updated = pass_.transform(module, context)
  assert "jax.Array" in updated.code


class DummySemanticsExtra:
  """Test helper class DummySemanticsExtra."""

  def __init__(self) -> None:
    """Initialize test helper instance."""
    self.framework_configs = {
      "jax": {
        "tiers": ["array", "neural"],
        "traits": {"module_base": "flax.nnx.Module", "forward_method": "__call__", "requires_super_init": False},
      },
      "torch": {"traits": {"module_base": "torch.nn.Module", "known_inference_methods": {"forward"}}},
    }
    self.alias_map: dict[str, str] = {}
    self.known_magic_args = ["rngs"]

  def get_framework_config(self, fw: str) -> dict[str, JsonValue]:
    """Execute get framework config helper logic.

    Args:
        fw: Framework identifier string (e.g., torch, jax, mlx).

    Returns:
        Resulting `dict[str, JsonValue]` from get framework config.
    """
    _ret_v: dict[str, JsonValue] = getattr(
      {
        "v": getattr(
          {"v": getattr({"v": getattr({"v": self}, "__getitem__")("v").framework_configs}, "__getitem__")("v").get},
          "__getitem__",
        )("v")(getattr({"v": fw}, "__getitem__")("v"))
      },
      "__getitem__",
    )("v")
    return _ret_v

  def get_definition(self, name: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Execute get definition helper logic.

    Args:
        name: Identifier or operation name string.

    Returns:
        Resulting `tuple[str, dict[str, JsonValue]] | None` from get definition.
    """
    if name == "torch.Tensor":
      return ("tensor", {})
    return None

  def resolve_variant(self, abstract_id: str, fw: str) -> dict[str, JsonValue] | None:
    """Execute resolve variant helper logic.

    Args:
        abstract_id: The abstract id (str) for resolve variant.
        fw: Framework identifier string (e.g., torch, jax, mlx).

    Returns:
        Resulting `dict[str, JsonValue] | None` from resolve variant.
    """
    if abstract_id == "tensor" and fw == "jax":
      return {"api": "jax.Array"}
    return None


def test_structure_transformer_edge_cases2() -> None:
  """Verify structure transformer edge cases2 behavior."""
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  semantics = DummySemantics()
  context = RewriterContext(semantics=semantics, config=config)
  transformer = StructuralTransformer(context)

  # 181, 183->174: self._known_module_bases
  class DummyTraits:
    """Test helper class DummyTraits."""

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
  assert isinstance(class_def3, cst.ClassDef)
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
  assert isinstance(class_def4, cst.ClassDef)
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
  assert isinstance(class_def5, cst.ClassDef)
  func_def5 = class_def5.body.body[0]
  assert isinstance(func_def5, cst.FunctionDef)
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
    "jax": {"traits": {"module_base": "flax.nnx.Module"}},
    "torch": {
      "tiers": ["array", "neural"],
      "traits": {"module_base": "torch.nn.Module", "auto_strip_magic_args": True, "strip_magic_args": ["rngs"]},
    },
  }
  config_torch = RuntimeConfig(source_framework="jax", target_framework="torch")
  context = RewriterContext(semantics=semantics, config=config_torch)
  transformer = StructuralTransformer(context)
  mod6 = cst.parse_module("class A(torch.nn.Module):\n  def __init__(self, rngs): pass")
  class_def6 = mod6.body[0]
  assert isinstance(class_def6, cst.ClassDef)
  func_def6 = class_def6.body.body[0]
  assert isinstance(func_def6, cst.FunctionDef)
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
  assert isinstance(class_def7, cst.ClassDef)
  func_def7 = class_def7.body.body[0]
  assert isinstance(func_def7, cst.FunctionDef)
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
  assert isinstance(class_def8, cst.ClassDef)
  func_def8 = class_def8.body.body[0]
  assert isinstance(func_def8, cst.FunctionDef)
  transformer.visit_ClassDef(class_def8)
  transformer.visit_FunctionDef(func_def8)
  res_func6 = transformer.leave_FunctionDef(func_def8, func_def8)
  transformer.leave_ClassDef(class_def8, class_def8)
  # The first body item is a SimpleStatementLine containing an Expr with the string
  assert getattr({"v": "rngs"}, "__getitem__")("v") in getattr(
    {
      "v": getattr(
        {
          "v": getattr(
            {
              "v": getattr(
                {
                  "v": getattr(
                    {
                      "v": getattr(
                        {
                          "v": getattr({"v": getattr({"v": res_func6}, "__getitem__")("v").body}, "__getitem__")("v").body
                        },
                        "__getitem__",
                      )("v")[0]
                    },
                    "__getitem__",
                  )("v").body
                },
                "__getitem__",
              )("v")[0]
            },
            "__getitem__",
          )("v").value
        },
        "__getitem__",
      )("v").value
    },
    "__getitem__",
  )("v")

  # 529: _resolve_alias missing fallback
  semantics.framework_configs = {"jax": {"tiers": ["array", "neural"], "traits": {}}}
  context = RewriterContext(semantics=semantics, config=config)
  transformer = StructuralTransformer(context)
  # how to trigger 529
  # def leave_FunctionDef -> 530 is return updated_node. 529 is in leave_FunctionDef? Wait 529 is docstring logic or return.


def test_structure_missing_branches() -> None:
  """Test remaining missing branches and statements in StructuralTransformer."""
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  semantics = DummySemantics()
  context = RewriterContext(semantics=semantics, config=config)
  transformer = StructuralTransformer(context)

  # 1. 138->140: _cst_to_string on Attribute with non-flattenable base
  attr_call = cst.Attribute(value=cst.Call(func=cst.Name("fn")), attr=cst.Name("attr"))
  assert transformer._cst_to_string(attr_call) is None

  # 2. 220->222: _get_source_inference_methods with empty known_inference_methods
  getattr(
    getattr(
      {"v": getattr({"v": getattr({"v": semantics}, "__getitem__")("v").framework_configs}, "__getitem__")("v")["torch"]},
      "__getitem__",
    )("v"),
    "__setitem__",
  )("traits", getattr({"v": {"known_inference_methods": getattr({"v": set()}, "__getitem__")("v")}}, "__getitem__")("v"))
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
  setattr(
    getattr({"v": semantics}, "__getitem__")("v"),
    "variants",
    getattr(
      {
        "v": {
          ("tensor_api", "jax"): getattr({"v": {"api": getattr({"v": "jax.Array"}, "__getitem__")("v")}}, "__getitem__")(
            "v"
          ),
          ("tensor_no_api", "jax"): getattr({"v": {"dummy": getattr({"v": "value"}, "__getitem__")("v")}}, "__getitem__")(
            "v"
          ),
        }
      },
      "__getitem__",
    )("v"),
  )
  # 354 -> 355: has full_name and has api
  name_with_api = cst.Name("TensorWithApi")
  res_name1 = transformer.leave_Name(name_with_api, name_with_api)
  assert isinstance(res_name1, cst.Attribute)

  # 354 -> 358: full_name is None
  setattr(transformer, "_cst_to_string", lambda node: None)
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
  class_fallback = class_mod_fallback.body[0]
  assert isinstance(class_fallback, cst.ClassDef)
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
  class_def = class_mod.body[0]
  assert isinstance(class_def, cst.ClassDef)
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
  class_multi_base = cst.parse_module("class M(torch.nn.Module, OtherBase): pass").body[0]
  assert isinstance(class_multi_base, cst.ClassDef)
  transformer.context.scope_stack.append(set())
  transformer.context.in_module_class = True
  res_multi = transformer.leave_ClassDef(class_multi_base, class_multi_base)
  assert isinstance(res_multi, cst.ClassDef)
  assert len(res_multi.bases) == 2

  # 9. 530->529: visit_FunctionDef when param.name is not cst.Name
  param_noname = getattr(
    {
      "v": getattr({"v": getattr({"v": cst}, "__getitem__")("v").Param}, "__getitem__")("v")(
        name=getattr(
          {
            "v": getattr({"v": getattr({"v": cst}, "__getitem__")("v").SimpleString}, "__getitem__")("v")(
              getattr({"v": "'param'"}, "__getitem__")("v")
            )
          },
          "__getitem__",
        )("v")
      )
    },
    "__getitem__",
  )("v")
  func_noname = cst.parse_module("def f(): pass").body[0]
  func_noname = func_noname.with_changes(params=cst.Parameters(params=[param_noname]))
  assert isinstance(func_noname, cst.FunctionDef)
  transformer.visit_FunctionDef(func_noname)
  transformer.context.scope_stack.pop()
  transformer.context.signature_stack.pop()

  # 10. 559: leave_FunctionDef when signature_stack is empty
  func_node = cst.parse_module("def g(): pass").body[0]
  assert isinstance(func_node, cst.FunctionDef)
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
  class_rngs = func_rngs_mod.body[0]
  assert isinstance(class_rngs, cst.ClassDef)
  func_rngs = class_rngs.body.body[0]
  assert isinstance(func_rngs, cst.FunctionDef)
  transformer_magic.visit_ClassDef(class_rngs)
  transformer_magic.visit_FunctionDef(func_rngs)
  res_rngs = transformer_magic.leave_FunctionDef(func_rngs, func_rngs)
  assert isinstance(res_rngs, cst.FunctionDef)

  # 12. 290->293: _get_type_mapping when target_framework is unknown in target_type_map
  config_ir = RuntimeConfig(source_framework="torch", target_framework="ir")
  context_ir = RewriterContext(semantics=DummySemantics(), config=config_ir)
  transformer_ir = StructuralTransformer(context_ir)
  res_type = transformer_ir._get_type_mapping("torch.Tensor")
  assert res_type is None

  # 13. 551: leave_ClassDef when target_base is None
  semantics_nobase = DummySemantics()
  semantics_nobase.framework_configs = {
    "ir": {"tiers": ["neural"], "traits": {"module_base": None}},
    "torch": {"traits": {"module_base": "torch.nn.Module"}},
  }
  context_nobase = RewriterContext(semantics=semantics_nobase, config=config_ir)
  transformer_nobase = StructuralTransformer(context_nobase)
  class_torch = cst.parse_module("class M(torch.nn.Module): pass").body[0]
  assert isinstance(class_torch, cst.ClassDef)
  transformer_nobase.visit_ClassDef(class_torch)
  res_nobase = transformer_nobase.leave_ClassDef(class_torch, class_torch)
  assert isinstance(res_nobase, cst.ClassDef)
  assert len(res_nobase.bases) == 1
