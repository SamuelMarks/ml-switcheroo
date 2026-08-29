"""Test suite for the Structure module."""

import typing

import libcst as cst
import pytest
from ml_switcheroo_ir.schema.ghost import SemanticTier

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.core.rewriter.passes.structure import StructuralPass, StructuralTransformer
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.semantics.schema import StructuralTraits


class MockSemantics(SemanticsManager):
  """Docstring."""

  def __init__(self) -> None:
    """Initializes the MockSemantics instance."""
    self.data: dict[str, typing.Any] = {}
    self.framework_configs: dict[str, typing.Any] = {
      "torch": {"traits": {"module_base": "torch.nn.Module", "forward_method": "forward"}},
      "jax": {
        "traits": {
          "module_base": "flax.nnx.Module",
          "forward_method": "__call__",
          "init_method_name": "__init__",
          "inject_magic_args": [("rngs", "nnx.Rngs")],
          "requires_super_init": False,
        },
        "alias": {"module": "flax.nnx", "name": "nnx"},
      },
    }
    self.data["Tensor"] = {"variants": {"jax": {"api": "jax.Array"}}}
    self._reverse_index: dict[str, tuple[str, dict[str, typing.Any]]] = {"torch.Tensor": ("Tensor", self.data["Tensor"])}

  def get_framework_config(self, fw: str) -> dict[str, typing.Any]:
    """Mock implementation of get framework configuration."""
    return self.framework_configs.get(fw, {})

  def get_definition(self, name: str) -> typing.Optional[tuple[str, dict[str, typing.Any]]]:
    """Mock implementation of get definition."""
    return self._reverse_index.get(name)

  def resolve_variant(self, aid: str, fw: str) -> typing.Any:
    """Mock implementation of resolve variant."""
    if aid in self.data and fw in self.data[aid].get("variants", {}):
      return self.data[aid]["variants"][fw]
    return None


@pytest.fixture
def run_pass() -> typing.Callable[[str], str]:
  """Docstring."""
  semantics = MockSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  context = RewriterContext(semantics, config)
  context.alias_map["torch"] = "torch"
  context.alias_map["torch.nn"] = "torch.nn"

  def _transform(code: str) -> str:
    """Helper to  transform."""
    module = cst.parse_module(code)
    struct_pass = StructuralPass()
    return typing.cast(str, struct_pass.transform(module, context).code)

  return _transform


def test_class_base_rewrite(run_pass: typing.Callable[[str], str]) -> None:
  """Verifies the behavior of class base rewrite."""
  code: str = "class Net(torch.nn.Module): pass"
  res: str = run_pass(code)
  assert "class Net(flax.nnx.Module):" in res


def test_class_base_rewrite_aliased(run_pass: typing.Callable[[str], str]) -> None:
  """Verifies the behavior of class base rewrite aliased."""
  code: str = "class Net(torch.nn.Module): pass"
  res: str = run_pass(code)
  assert "flax.nnx.Module" in res


def test_method_renaming(run_pass: typing.Callable[[str], str]) -> None:
  """Verifies the behavior of method renaming."""
  code: str = "\nclass Net(torch.nn.Module):\n    def forward(self, x): pass\n"
  res: str = run_pass(code)
  assert "def __call__(self, x):" in res


def test_magic_arg_injection(run_pass: typing.Callable[[str], str]) -> None:
  """Verifies the behavior of magic argument injection."""
  code: str = "\nclass Net(torch.nn.Module):\n    def __init__(self, dim): pass\n"
  res: str = run_pass(code)
  assert "def __init__(self, rngs: nnx.Rngs, dim):" in res


def test_super_init_stripping(run_pass: typing.Callable[[str], str]) -> None:
  """Verifies the behavior of super initialization stripping."""
  code: str = "\nclass Net(torch.nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.x = 1\n"
  res: str = run_pass(code)
  assert "super().__init__()" not in res
  assert "self.x = 1" in res


def test_type_hint_rewrite(run_pass: typing.Callable[[str], str]) -> None:
  """Verifies the behavior of type hint rewrite."""
  code: str = "def f(x: torch.Tensor): pass"
  res: str = run_pass(code)
  assert "x: jax.Array" in res


def test_ignore_non_module_classes(run_pass: typing.Callable[[str], str]) -> None:
  """Verifies the behavior of ignore non module classes."""
  code: str = "\nclass Data:\n    def forward(self): pass\n"
  res: str = run_pass(code)
  assert "class Data:" in res
  assert "def forward(self):" in res


# --- Merged from test_structure_extra.py ---


class DummySemantics(SemanticsManager):
  """Docstring."""

  def __init__(self) -> None:
    """Initializes the DummySemantics instance."""
    self.configs: dict[str, typing.Any] = {}
    self.framework_configs = self.configs
    self.definitions: dict[str, typing.Any] = {}
    self.variants: dict[tuple[str, str], typing.Any] = {}
    self.verified = True
    self.known_magic_args: set[str] = set()

  def get_framework_config(self, fw: str) -> dict[str, typing.Any]:
    """Mock implementation of get framework configuration."""
    return self.configs.get(fw, {})

  def get_definition(self, name: str) -> typing.Optional[tuple[str, dict[str, typing.Any]]]:
    """Mock implementation of get definition."""
    return self.definitions.get(name)

  def resolve_variant(self, abstract_id: str, fw: str) -> typing.Any:
    """Mock implementation of resolve variant."""
    return self.variants.get((abstract_id, fw))


def get_transformer() -> tuple[StructuralTransformer, DummySemantics, RewriterContext]:
  """Gets transformer."""
  semantics = DummySemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=True)
  ctx = RewriterContext(semantics, config)
  transformer = StructuralTransformer(ctx)
  return (transformer, semantics, ctx)


def test_target_traits_fallback() -> None:
  """Verifies the behavior of target traits fallback."""
  (transformer, sem, ctx) = get_transformer()
  traits = transformer.target_traits
  assert isinstance(traits, StructuralTraits)
  assert transformer._cached_target_traits is traits


def test_get_target_tiers_fallback() -> None:
  """Gets target tiers fallback."""
  (transformer, sem, ctx) = get_transformer()
  tiers = transformer._get_target_tiers()
  assert SemanticTier.ARRAY_API.value in tiers


def test_cst_to_string_fallback() -> None:
  """Verifies the behavior of cst to string fallback."""
  (transformer, sem, ctx) = get_transformer()
  node = cst.BinaryOperation(left=cst.Name("a"), operator=cst.Add(), right=cst.Name("b"))
  assert transformer._cst_to_string(node) is None
  assert transformer._get_qualified_name(node) is None


def test_is_framework_base_empty() -> None:
  """Checks if is framework base empty."""
  (transformer, sem, ctx) = get_transformer()
  assert transformer._is_framework_base("") is False
  assert transformer._is_framework_base(None) is False  # type: ignore


def test_is_framework_base_traits_object() -> None:
  """Checks if is framework base traits object."""
  (transformer, sem, ctx) = get_transformer()

  class DummyTraits:
    module_base = "my.Framework"

  sem.configs["torch"] = {"traits": DummyTraits()}
  assert transformer._is_framework_base("my.Framework") is True


def test_is_framework_base_suffix() -> None:
  """Checks if is framework base suffix."""
  (transformer, sem, ctx) = get_transformer()
  sem.configs["torch"] = {"traits": {"module_base": "torch.nn.Module"}}
  assert transformer._is_framework_base("nn.Module") is True
  assert transformer._is_framework_base("other.Module") is False


def test_get_source_inference_methods_fallback() -> None:
  """Gets source inference methods fallback."""
  (transformer, sem, ctx) = get_transformer()
  methods = transformer._get_source_inference_methods()
  assert "forward" in methods


def test_leave_name_not_in_annotation() -> None:
  """Verifies the behavior of leave name not in annotation."""
  (transformer, sem, ctx) = get_transformer()
  name = cst.Name("x")
  new_name: typing.Any = transformer.leave_Name(name, name)
  assert new_name is name


def test_leave_attribute_not_in_annotation() -> None:
  """Verifies the behavior of leave attribute not in annotation."""
  (transformer, sem, ctx) = get_transformer()
  attr = cst.Attribute(value=cst.Name("x"), attr=cst.Name("y"))
  new_attr: typing.Any = transformer.leave_Attribute(attr, attr)
  assert new_attr is attr


def test_visit_classdef_fallback_and_error() -> None:
  """Verifies the behavior of visit classdef fallback and correctly handling an error."""
  (transformer, sem, ctx) = get_transformer()
  sem.configs["torch"] = {"traits": {"module_base": "torch.nn.Module"}}
  class_node = typing.cast(cst.ClassDef, cst.parse_module("class Net(nn.Module): pass").body[0])
  transformer.visit_ClassDef(class_node)
  assert ctx.in_module_class
  sem.configs["jax"] = {"tiers": ["array_api"]}
  ctx.current_stmt_errors.clear()
  transformer.visit_ClassDef(class_node)
  assert "does not support Neural Network classes" in ctx.current_stmt_errors[0]
  res: typing.Any = transformer.leave_ClassDef(class_node, class_node)
  assert isinstance(res, cst.FlattenSentinel)
  assert not ctx.in_module_class


def test_leave_classdef_unmapped_base() -> None:
  """Verifies the behavior of leave classdef unmapped base."""
  (transformer, sem, ctx) = get_transformer()
  sem.configs["torch"] = {"traits": {"module_base": "torch.nn.Module"}}
  sem.configs["jax"] = {"traits": {"module_base": "flax.nnx.Module"}}
  class_node = typing.cast(cst.ClassDef, cst.parse_module("class Net(nn.Module, Other): pass").body[0])
  transformer.visit_ClassDef(class_node)
  new_node: typing.Any = transformer.leave_ClassDef(class_node, class_node)
  assert "flax.nnx.Module" in transformer._cst_to_string(new_node.bases[0].value)  # type: ignore
  assert "Other" in transformer._cst_to_string(new_node.bases[1].value)  # type: ignore


def test_leave_functiondef_no_stack() -> None:
  """Verifies the behavior of leave functiondef no stack."""
  (transformer, sem, ctx) = get_transformer()
  func = typing.cast(cst.FunctionDef, cst.parse_module("def foo(): pass").body[0])
  assert transformer.leave_FunctionDef(func, func) is func


def test_leave_functiondef_renaming() -> None:
  """Verifies the behavior of leave functiondef renaming."""
  (transformer, sem, ctx) = get_transformer()
  sem.configs["jax"] = {"traits": {"init_method_name": "setup"}}
  func = typing.cast(cst.FunctionDef, cst.parse_module("def __init__(self): pass").body[0])
  transformer.visit_FunctionDef(func)
  ctx.in_module_class = True
  ctx.signature_stack[-1].is_module_method = True
  new_func: typing.Any = transformer.leave_FunctionDef(func, func)
  assert new_func.name.value == "setup"


def test_leave_functiondef_magic_args() -> None:
  """Verifies the behavior of leave functiondef magic arguments."""
  (transformer, sem, ctx) = get_transformer()
  sem.known_magic_args.add("rngs")
  sem.configs["jax"] = {
    "traits": {"auto_strip_magic_args": True, "strip_magic_args": ["ctx"], "inject_magic_args": [("rngs", "int")]}
  }
  func = typing.cast(cst.FunctionDef, cst.parse_module("def __init__(self, ctx, rngs): pass").body[0])
  transformer.visit_FunctionDef(func)
  ctx.in_module_class = True
  ctx.signature_stack[-1].is_module_method = True
  new_func: typing.Any = transformer.leave_FunctionDef(func, func)
  params: list[str] = [p.name.value for p in new_func.params.params if isinstance(p.name, cst.Name)]
  assert "ctx" not in params
  assert "rngs" in params


def test_super_init_logic() -> None:
  """Verifies the behavior of super initialization logic."""
  (transformer, sem, ctx) = get_transformer()
  sem.configs["jax"] = {"traits": {"requires_super_init": True}}
  func = typing.cast(cst.FunctionDef, cst.parse_module("def __init__(self): pass").body[0])
  transformer.visit_FunctionDef(func)
  ctx.in_module_class = True
  ctx.signature_stack[-1].is_module_method = True
  new_func: typing.Any = transformer.leave_FunctionDef(func, func)
  code = cst.Module([new_func]).code
  assert "super().__init__()" in code
  sem.configs["jax"] = {"traits": {"requires_super_init": False}}
  func2 = typing.cast(cst.FunctionDef, cst.parse_module("def __init__(self): pass").body[0])
  func2 = func2.with_changes(body=cst.SimpleStatementSuite(body=[cst.Pass()]))
  res: typing.Any = transformer._strip_super_init(func2)
  assert isinstance(res.body, cst.SimpleStatementSuite)


def test_leave_module_preamble_no_cover() -> None:
  """Docstring."""
  from ml_switcheroo.config import RuntimeConfig
  from ml_switcheroo.core.rewriter.context import RewriterContext
  from ml_switcheroo.core.rewriter.passes.structure import StructuralTransformer
  from ml_switcheroo.semantics.manager import SemanticsManager

  ctx = RewriterContext(semantics=SemanticsManager(), config=RuntimeConfig())
  transformer = StructuralTransformer(ctx)
  mod = cst.parse_module("a = 1")
  ctx.module_preamble.append("import sys")
  new_mod: typing.Any = transformer.leave_Module(mod, mod)
  assert "import sys" in new_mod.code


def test_leave_attribute_super_shim() -> None:
  """Docstring."""
  from ml_switcheroo.config import RuntimeConfig
  from ml_switcheroo.core.rewriter.context import RewriterContext
  from ml_switcheroo.core.rewriter.passes.structure import StructuralTransformer
  from ml_switcheroo.semantics.manager import SemanticsManager

  ctx = RewriterContext(semantics=SemanticsManager(), config=RuntimeConfig())

  class SuperShim(cst.CSTTransformer):
    def leave_Attribute(self, original: typing.Any, updated: typing.Any) -> typing.Any:
      return updated.with_changes(value=cst.Name("shimmed"))

  class ShimmedStructuralPass(StructuralTransformer, SuperShim):  # type: ignore
    pass

  transformer = ShimmedStructuralPass(ctx)  # type: ignore
  mod = cst.parse_module("a.b")
  attr = typing.cast(cst.Expr, typing.cast(cst.SimpleStatementLine, mod.body[0]).body[0]).value

  # Needs to bypass our custom leave_Attribute logic
  attr_updated: typing.Any = transformer.leave_Attribute(attr, attr)
  assert typing.cast(cst.Name, attr_updated.value).value == "shimmed"


# --- Merged from test_structure_extra2.py ---


class DummySemantics:
  """Docstring."""

  def __init__(self) -> None:
    """Initializes the DummySemantics instance."""
    self.definitions: dict[str, typing.Any] = {}
    self.configs: dict[str, typing.Any] = {}
    self.framework_configs = self.configs
    self.variants: dict[tuple[str, str], typing.Any] = {}
    self.known_magic_args: set[str] = set()

  def resolve_definition(self, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Mock implementation of resolve definition."""
    return None

  def get_standard_module(self, *args: typing.Any, **kwargs: typing.Any) -> str:
    """Mock implementation of get standard module."""
    return "nn"

  def resolve_variant(self, op_id: str, fw: str) -> typing.Any:
    """Mock implementation of resolve variant."""
    return self.variants.get((op_id, fw))

  def get_framework_config(self, framework: str) -> dict[str, typing.Any]:
    """Mock implementation of get framework configuration."""
    return self.configs.get(framework, {})

  def get_definition(self, api: str) -> typing.Optional[tuple[str, dict[str, typing.Any]]]:
    """Mock implementation of get definition."""
    return self.definitions.get(api)


def get_transformer_extra() -> tuple[StructuralTransformer, DummySemantics, RewriterContext]:
  """Gets transformer."""
  sem = DummySemantics()
  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  ctx = RewriterContext(sem, cfg)  # type: ignore
  return (StructuralTransformer(ctx), sem, ctx)


def test_preamble_and_docstring() -> None:
  """Docstring."""
  (transformer, sem, ctx) = get_transformer()
  func = typing.cast(cst.FunctionDef, cst.parse_module('def __init__(self):\n  """doc"""\n  pass').body[0])
  transformer.visit_FunctionDef(func)
  ctx.in_module_class = True
  ctx.signature_stack[-1].is_module_method = True
  ctx.signature_stack[-1].injected_args.append(("y", "int"))
  ctx.signature_stack[-1].preamble_stmts.append("print(1)")
  ctx.signature_stack[-1].preamble_stmts.append("1 = 2")
  new_func: typing.Any = transformer.leave_FunctionDef(func, func)
  code = cst.Module([new_func]).code
  assert "print(1)" in code
  assert "y: Injected." in code
  assert "1 = 2" not in code
  func_simple = typing.cast(cst.FunctionDef, cst.parse_module("def foo(): print(2)").body[0])
  res_simple: typing.Any = transformer._convert_to_indented_block(func_simple)
  assert isinstance(res_simple.body, cst.IndentedBlock)


def test_strip_argument_from_signature() -> None:
  """Verifies the behavior of strip argument from signature."""
  (transformer, sem, ctx) = get_transformer()
  func = typing.cast(cst.FunctionDef, cst.parse_module("def f(x, y): pass").body[0])
  res: typing.Any = transformer._strip_argument_from_signature(func, "x")
  assert res.params.params[0].name.value == "y"


def test_fix_comma() -> None:
  """Fixes comma."""
  (transformer, sem, ctx) = get_transformer()
  func = typing.cast(cst.FunctionDef, cst.parse_module("def f(x, y): pass").body[0])
  params = list(func.params.params)
  params[-1] = params[-1].with_changes(comma=cst.Comma())
  res: typing.Any = transformer._fix_comma(func, params)
  assert res.params.params[-1].comma == cst.MaybeSentinel.DEFAULT


def test_leave_module_preamble_empty_stmts() -> None:
  """Verifies the behavior of leave module preamble empty stmts."""
  (transformer, sem, ctx) = get_transformer()
  mod = cst.parse_module("a = 1")
  ctx.module_preamble.append("invalid code ###")
  new_mod: typing.Any = transformer.leave_Module(mod, mod)
  assert "invalid code" not in new_mod.code
  assert new_mod is mod


def test_leave_name_in_annotation_success() -> None:
  """Verifies the behavior of leave name in annotation successfully."""
  (transformer, sem, ctx) = get_transformer()
  transformer._in_annotation = True
  sem.definitions["UnknownType"] = ("UnknownType", {})
  sem.variants["UnknownType", "jax"] = {"api": "jnp.MappedType"}
  name = cst.Name("UnknownType")
  res: typing.Any = transformer.leave_Name(name, name)
  assert transformer._cst_to_string(res) == "jnp.MappedType"


def test_leave_attribute_in_annotation_success() -> None:
  """Verifies the behavior of leave attribute in annotation successfully."""
  (transformer, sem, ctx) = get_transformer()
  transformer._in_annotation = True
  sem.definitions["Unknown.Type"] = ("UnknownType", {})
  sem.variants["UnknownType", "jax"] = {"api": "jnp.MappedType"}
  attr = cst.Attribute(value=cst.Name("Unknown"), attr=cst.Name("Type"))
  res: typing.Any = transformer.leave_Attribute(attr, attr)
  assert transformer._cst_to_string(res) == "jnp.MappedType"


def test_leave_classdef_raw_name_fallback() -> None:
  """Verifies the behavior of leave classdef raw name fallback."""
  (transformer, sem, ctx) = get_transformer()
  sem.configs["torch"] = {"traits": {"module_base": "torch.nn.Module"}}
  sem.configs["jax"] = {"traits": {"module_base": "flax.nnx.Module"}}
  class_node = typing.cast(cst.ClassDef, cst.parse_module("class Net(nn.Module): pass").body[0])
  transformer.visit_ClassDef(class_node)
  orig_gqn = transformer._get_qualified_name
  transformer._get_qualified_name = lambda n: None  # type: ignore
  new_node: typing.Any = transformer.leave_ClassDef(class_node, class_node)
  transformer._get_qualified_name = orig_gqn  # type: ignore
  assert "flax.nnx.Module" in transformer._cst_to_string(new_node.bases[0].value)  # type: ignore


def test_convert_to_indented_block_fallback() -> None:
  """Converts to indented block fallback."""
  (transformer, sem, ctx) = get_transformer()
  func = typing.cast(cst.FunctionDef, cst.parse_module("def foo():\n  pass").body[0])
  assert transformer._convert_to_indented_block(func) is func


def test_ensure_super_init_already_has() -> None:
  """Verifies the behavior of ensure super initialization already has."""
  (transformer, sem, ctx) = get_transformer()
  func = typing.cast(cst.FunctionDef, cst.parse_module("def __init__(self):\n  super().__init__()").body[0])
  assert transformer._ensure_super_init(func) is func


def test_strip_super_init_no_body() -> None:
  """Verifies the behavior of strip super initialization no body."""
  (transformer, sem, ctx) = get_transformer()
  func = typing.cast(cst.FunctionDef, cst.parse_module("def f(): pass").body[0])
  func = func.with_changes(body=cst.Pass())
  assert transformer._strip_super_init(func) is func


def test_has_super_init_false() -> None:
  """Checks if has super initialization false."""
  (transformer, sem, ctx) = get_transformer()
  func = typing.cast(cst.FunctionDef, cst.parse_module("def __init__(self):\n  pass").body[0])
  assert transformer._has_super_init(func) is False


def test_update_docstring_fallback() -> None:
  """Docstring."""
  (transformer, sem, ctx) = get_transformer()
  func0 = typing.cast(cst.FunctionDef, cst.parse_module("def f(): pass").body[0])
  func0 = func0.with_changes(body=cst.SimpleStatementSuite(body=[]))
  assert transformer._update_docstring(func0, [("a", "b")]) is func0
  func = typing.cast(cst.FunctionDef, cst.parse_module("def f():\n  x = 1").body[0])
  assert transformer._update_docstring(func, [("a", "b")]) is func
  func2 = typing.cast(cst.FunctionDef, cst.parse_module("def f():\n  'doc'").body[0])
  assert transformer._update_docstring(func2, [("a", "b")]) is func2


def test_visit_classdef_raw_name_fallback() -> None:
  """Verifies the behavior of visit classdef raw name fallback."""
  (transformer, sem, ctx) = get_transformer()
  sem.configs["torch"] = {"traits": {"module_base": "torch.nn.Module"}}
  class_node = typing.cast(cst.ClassDef, cst.parse_module("class Net(nn.Module): pass").body[0])
  orig_gqn = transformer._get_qualified_name
  transformer._get_qualified_name = lambda n: None  # type: ignore
  transformer.visit_ClassDef(class_node)
  transformer._get_qualified_name = orig_gqn  # type: ignore
  assert ctx.in_module_class


def test_leave_attribute_fallback_super() -> None:
  """Verifies the behavior of leave attribute fallback super."""
  (transformer, sem, ctx) = get_transformer()
  transformer._in_annotation = False
  attr = cst.Attribute(value=cst.Name("Unknown"), attr=cst.Name("Type"))
  res: typing.Any = transformer.leave_Attribute(attr, attr)
  assert res is attr
