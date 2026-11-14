"""Test suite for the Structure module."""

import typing

import libcst as cst
import pytest
from ml_switcheroo_ir.schema.ghost import SemanticTier
from pydantic import JsonValue

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.core.rewriter.passes.structure import StructuralPass, StructuralTransformer
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.semantics.schema import StructuralTraits


class MockSemantics(SemanticsManager):
  """Test double `MockSemantics` used in structure tests."""

  def __init__(self) -> None:
    """Initializes the MockSemantics instance."""
    super().__init__(load_defaults=False)
    self.framework_configs = {
      "torch": {"traits": {"module_base": "torch.nn.Module", "forward_method": "forward"}},
      "jax": {
        "traits": {
          "module_base": "flax.nnx.Module",
          "forward_method": "__call__",
          "init_method_name": "__init__",
          "inject_magic_args": [["rngs", "nnx.Rngs"]],
          "requires_super_init": False,
        },
        "alias": {"module": "flax.nnx", "name": "nnx"},
      },
    }
    self.data["Tensor"] = {"variants": {"jax": {"api": "jax.Array"}}}
    self._reverse_index = {"torch.Tensor": ("Tensor", self.data["Tensor"])}

  def get_framework_config(self, fw: str) -> dict[str, JsonValue]:
    """Mock implementation of get framework configuration.

    Args:
        fw: Framework identifier string (e.g., torch, jax, mlx).

    Returns:
        Resulting `dict[str, JsonValue]` from get framework config.
    """
    return self.framework_configs.get(fw, {})

  def get_definition(self, name: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Mock implementation of get definition.

    Args:
        name: Identifier or operation name string.

    Returns:
        Resulting `tuple[str, dict[str, JsonValue]] | None` from get definition.
    """
    return self._reverse_index.get(name)

  def resolve_variant(self, aid: str, fw: str) -> dict[str, JsonValue] | None:
    """Mock implementation of resolve variant.

    Args:
        aid: Abstract operation identifier string.
        fw: Framework identifier string (e.g., torch, jax, mlx).

    Returns:
        Resulting `dict[str, JsonValue] | None` from resolve variant.
    """
    op_entry = self.data.get(aid)
    if op_entry is None:
      return None
    variants = op_entry.get("variants")
    if not isinstance(variants, dict):
      return None
    fw_variant = variants.get(fw)
    return fw_variant if isinstance(fw_variant, dict) else None


@pytest.fixture
def run_pass() -> typing.Callable[[str], str]:
  """Create a configured `typing.Callable[[str], str]` fixture for structure tests.

  Returns:
      Configured `run_pass` fixture value (typing.Callable[[str], str]).
  """
  semantics = MockSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  context = RewriterContext(semantics, config)
  context.alias_map["torch"] = "torch"
  context.alias_map["torch.nn"] = "torch.nn"

  def _transform(code: str) -> str:
    """Execute `StructuralPass` to support transform in structure tests.

    Args:
        code: Python or DSL source code string to parse or transpile.

    Returns:
        Resulting `str` from transform.
    """
    module = cst.parse_module(code)
    struct_pass = StructuralPass()
    return struct_pass.transform(module, context).code

  return _transform


def test_class_base_rewrite(run_pass: typing.Callable[[str], str]) -> None:
  """Verify `run_pass` emits `class Net(flax.nnx.Module):` for class base rewrite.

  Args:
      run_pass: Input `run_pass` parameter value (`typing.Callable[[str], str]`).
  """
  code: str = "class Net(torch.nn.Module): pass"
  res: str = run_pass(code)
  assert "class Net(flax.nnx.Module):" in res


def test_class_base_rewrite_aliased(run_pass: typing.Callable[[str], str]) -> None:
  """Verify `run_pass` emits `flax.nnx.Module` for class base rewrite aliased.

  Args:
      run_pass: Input `run_pass` parameter value (`typing.Callable[[str], str]`).
  """
  code: str = "class Net(torch.nn.Module): pass"
  res: str = run_pass(code)
  assert "flax.nnx.Module" in res


def test_method_renaming(run_pass: typing.Callable[[str], str]) -> None:
  """Verify `run_pass` emits `def __call__(self, x):` for method renaming.

  Args:
      run_pass: Input `run_pass` parameter value (`typing.Callable[[str], str]`).
  """
  code: str = "\nclass Net(torch.nn.Module):\n    def forward(self, x): pass\n"
  res: str = run_pass(code)
  assert "def __call__(self, x):" in res


def test_magic_arg_injection(run_pass: typing.Callable[[str], str]) -> None:
  """Verify `run_pass` handles magic arg injection as expected.

  Args:
      run_pass: Input `run_pass` parameter value (`typing.Callable[[str], str]`).
  """
  code: str = "\nclass Net(torch.nn.Module):\n    def __init__(self, dim): pass\n"
  res: str = run_pass(code)
  assert "def __init__(self, rngs: nnx.Rngs, dim):" in res


def test_super_init_stripping(run_pass: typing.Callable[[str], str]) -> None:
  """Verify `run_pass` emits `self.x = 1` for super init stripping.

  Args:
      run_pass: Input `run_pass` parameter value (`typing.Callable[[str], str]`).
  """
  code: str = "\nclass Net(torch.nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.x = 1\n"
  res: str = run_pass(code)
  assert "super().__init__()" not in res
  assert "self.x = 1" in res


def test_type_hint_rewrite(run_pass: typing.Callable[[str], str]) -> None:
  """Verify `run_pass` emits `x: jax.Array` for type hint rewrite.

  Args:
      run_pass: Input `run_pass` parameter value (`typing.Callable[[str], str]`).
  """
  code: str = "def f(x: torch.Tensor): pass"
  res: str = run_pass(code)
  assert "x: jax.Array" in res


def test_ignore_non_module_classes(run_pass: typing.Callable[[str], str]) -> None:
  """Verify `run_pass` emits `class Data:` for ignore non module classes.

  Args:
      run_pass: Input `run_pass` parameter value (`typing.Callable[[str], str]`).
  """
  code: str = "\nclass Data:\n    def forward(self): pass\n"
  res: str = run_pass(code)
  assert "class Data:" in res
  assert "def forward(self):" in res


class DummySemantics(SemanticsManager):
  """Test double `DummySemantics` used in structure tests."""

  def __init__(self) -> None:
    """Initializes the DummySemantics instance."""
    super().__init__(load_defaults=False)
    self.configs: dict[str, dict[str, JsonValue]] = {}
    self.framework_configs = self.configs
    self.definitions: dict[str, tuple[str, dict[str, JsonValue]]] = {}
    self.variants: dict[tuple[str, str], dict[str, JsonValue]] = {}
    self.verified = True

  def get_framework_config(self, fw: str) -> dict[str, JsonValue]:
    """Mock implementation of get framework configuration.

    Args:
        fw: Framework identifier string (e.g., torch, jax, mlx).

    Returns:
        Resulting `dict[str, JsonValue]` from get framework config.
    """
    _ret_v: dict[str, JsonValue] = self.configs.get(fw, {})
    return _ret_v

  def get_definition(self, name: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Mock implementation of get definition.

    Args:
        name: Identifier or operation name string.

    Returns:
        Resulting `tuple[str, dict[str, JsonValue]] | None` from get definition.
    """
    _ret_v: tuple[str, dict[str, JsonValue]] | None = self.definitions.get(name)
    return _ret_v

  def resolve_variant(self, abstract_id: str, fw: str) -> dict[str, JsonValue] | None:
    """Mock implementation of resolve variant.

    Args:
        abstract_id: Canonical abstract operation identifier (`str`).
        fw: Framework identifier string (e.g., torch, jax, mlx).

    Returns:
        Resulting `dict[str, JsonValue] | None` from resolve variant.
    """
    _ret_v: dict[str, JsonValue] | None = self.variants.get((abstract_id, fw))
    return _ret_v


def get_transformer() -> tuple[StructuralTransformer, DummySemantics, RewriterContext]:
  """Gets transformer.

  Returns:
      Resulting `tuple[StructuralTransformer, DummySemantics, RewriterContext]` from get transformer.
  """
  semantics = DummySemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=True)
  ctx = RewriterContext(semantics, config)
  transformer = StructuralTransformer(ctx)
  return (transformer, semantics, ctx)


def test_target_traits_fallback() -> None:
  """Verify `get_transformer` handles target traits fallback as expected."""
  (transformer, _sem, _ctx) = get_transformer()
  traits = transformer.target_traits
  assert isinstance(traits, StructuralTraits)
  assert transformer._cached_target_traits is traits


def test_get_target_tiers_fallback() -> None:
  """Gets target tiers fallback."""
  (transformer, _sem, _ctx) = get_transformer()
  tiers = transformer._get_target_tiers()
  assert SemanticTier.ARRAY_API.value in tiers


def test_cst_to_string_fallback() -> None:
  """Verify `get_transformer` handles CST to string fallback as expected."""
  (transformer, _sem, _ctx) = get_transformer()
  node = cst.BinaryOperation(left=cst.Name("a"), operator=cst.Add(), right=cst.Name("b"))
  assert transformer._cst_to_string(node) is None
  assert transformer._get_qualified_name(node) is None


def test_is_framework_base_empty() -> None:
  """Checks if is framework base empty."""
  (transformer, _sem, _ctx) = get_transformer()
  assert transformer._is_framework_base("") is False


def test_is_framework_base_traits_object(monkeypatch: pytest.MonkeyPatch) -> None:
  """Checks if is framework base traits object.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  (transformer, sem, _ctx) = get_transformer()

  class DummyTraits:
    """Sentinel `DummyTraits` fixture class used in structure tests."""

    module_base = "my.Framework"

  monkeypatch.setattr(sem, "framework_configs", {"torch": {"traits": DummyTraits()}}, raising=False)
  assert transformer._is_framework_base("my.Framework") is True


def test_is_framework_base_suffix() -> None:
  """Checks if is framework base suffix."""
  (transformer, sem, _ctx) = get_transformer()
  sem.configs["torch"] = {"traits": {"module_base": "torch.nn.Module"}}
  assert transformer._is_framework_base("nn.Module") is True
  assert transformer._is_framework_base("other.Module") is False


def test_get_source_inference_methods_fallback() -> None:
  """Gets source inference methods fallback."""
  (transformer, _sem, _ctx) = get_transformer()
  methods = transformer._get_source_inference_methods()
  assert "forward" in methods


def test_leave_name_not_in_annotation() -> None:
  """Verify `get_transformer` handles leave name not in annotation as expected."""
  (transformer, _sem, _ctx) = get_transformer()
  name = cst.Name("x")
  new_name = transformer.leave_Name(name, name)
  assert new_name is name


def test_leave_attribute_not_in_annotation() -> None:
  """Verify `get_transformer` handles leave attribute not in annotation as expected."""
  (transformer, _sem, _ctx) = get_transformer()
  attr = cst.Attribute(value=cst.Name("x"), attr=cst.Name("y"))
  new_attr = transformer.leave_Attribute(attr, attr)
  assert new_attr is attr


def test_visit_classdef_fallback_and_error() -> None:
  """Verify `get_transformer` handles visit classdef fallback and error as expected."""
  (transformer, sem, ctx) = get_transformer()
  sem.configs["torch"] = {"traits": {"module_base": "torch.nn.Module"}}
  class_node = cst.ensure_type(cst.parse_module("class Net(nn.Module): pass").body[0], cst.ClassDef)
  transformer.visit_ClassDef(class_node)
  assert ctx.in_module_class
  sem.configs["jax"] = {"tiers": ["array_api"]}
  ctx.current_stmt_errors.clear()
  transformer.visit_ClassDef(class_node)
  assert "does not support Neural Network classes" in ctx.current_stmt_errors[0]
  res = transformer.leave_ClassDef(class_node, class_node)
  assert isinstance(res, cst.FlattenSentinel)
  assert not ctx.in_module_class


def test_leave_classdef_unmapped_base() -> None:
  """Verify `get_transformer` handles leave classdef unmapped base as expected."""
  (transformer, sem, _ctx) = get_transformer()
  sem.configs["torch"] = {"traits": {"module_base": "torch.nn.Module"}}
  sem.configs["jax"] = {"traits": {"module_base": "flax.nnx.Module"}}
  class_node = cst.ensure_type(cst.parse_module("class Net(nn.Module, Other): pass").body[0], cst.ClassDef)
  transformer.visit_ClassDef(class_node)
  new_node = cst.ensure_type(transformer.leave_ClassDef(class_node, class_node), cst.ClassDef)
  base0 = transformer._cst_to_string(new_node.bases[0].value)
  base1 = transformer._cst_to_string(new_node.bases[1].value)
  assert base0 is not None
  assert "flax.nnx.Module" in base0
  assert base1 is not None
  assert "Other" in base1


def test_leave_functiondef_no_stack() -> None:
  """Verify `get_transformer` handles leave functiondef no stack as expected."""
  (transformer, _sem, _ctx) = get_transformer()
  func = cst.ensure_type(cst.parse_module("def foo(): pass").body[0], cst.FunctionDef)
  assert transformer.leave_FunctionDef(func, func) is func


def test_leave_functiondef_renaming() -> None:
  """Verify `get_transformer` handles leave functiondef renaming as expected."""
  (transformer, sem, ctx) = get_transformer()
  sem.configs["jax"] = {"traits": {"init_method_name": "setup"}}
  func = cst.ensure_type(cst.parse_module("def __init__(self): pass").body[0], cst.FunctionDef)
  transformer.visit_FunctionDef(func)
  ctx.in_module_class = True
  ctx.signature_stack[-1].is_module_method = True
  new_func = transformer.leave_FunctionDef(func, func)
  assert new_func.name.value == "setup"


def test_leave_functiondef_magic_args() -> None:
  """Verify `get_transformer` emits `rngs` for leave functiondef magic args."""
  (transformer, sem, ctx) = get_transformer()
  sem.known_magic_args.add("rngs")
  sem.configs["jax"] = {
    "traits": {"auto_strip_magic_args": True, "strip_magic_args": ["ctx"], "inject_magic_args": [["rngs", "int"]]}
  }
  func = cst.ensure_type(cst.parse_module("def __init__(self, ctx, rngs): pass").body[0], cst.FunctionDef)
  transformer.visit_FunctionDef(func)
  ctx.in_module_class = True
  ctx.signature_stack[-1].is_module_method = True
  new_func = transformer.leave_FunctionDef(func, func)
  params: list[str] = [p.name.value for p in new_func.params.params]
  assert "ctx" not in params
  assert "rngs" in params


def test_super_init_logic() -> None:
  """Verify `get_transformer` emits `super().__init__()` for super init logic."""
  (transformer, sem, ctx) = get_transformer()
  sem.configs["jax"] = {"traits": {"requires_super_init": True}}
  func = cst.ensure_type(cst.parse_module("def __init__(self): pass").body[0], cst.FunctionDef)
  transformer.visit_FunctionDef(func)
  ctx.in_module_class = True
  ctx.signature_stack[-1].is_module_method = True
  new_func = transformer.leave_FunctionDef(func, func)
  code = cst.Module([new_func]).code
  assert "super().__init__()" in code
  sem.configs["jax"] = {"traits": {"requires_super_init": False}}
  func2 = cst.ensure_type(cst.parse_module("def __init__(self): pass").body[0], cst.FunctionDef)
  func2 = func2.with_changes(body=cst.SimpleStatementSuite(body=[cst.Pass()]))
  res = transformer._strip_super_init(func2)
  assert isinstance(res.body, cst.SimpleStatementSuite)


def test_leave_module_preamble_no_cover() -> None:
  """Verify `RewriterContext` emits `import sys` for leave module preamble no cover."""
  from ml_switcheroo.config import RuntimeConfig
  from ml_switcheroo.core.rewriter.context import RewriterContext
  from ml_switcheroo.core.rewriter.passes.structure import StructuralTransformer
  from ml_switcheroo.semantics.manager import SemanticsManager

  ctx = RewriterContext(semantics=SemanticsManager(), config=RuntimeConfig())
  transformer = StructuralTransformer(ctx)
  mod = cst.parse_module("a = 1")
  ctx.module_preamble.append("import sys")
  new_mod = transformer.leave_Module(mod, mod)
  assert "import sys" in new_mod.code


def test_leave_attribute_super_shim() -> None:
  """Verify `RewriterContext` handles leave attribute super shim as expected."""
  from ml_switcheroo.config import RuntimeConfig
  from ml_switcheroo.core.rewriter.context import RewriterContext
  from ml_switcheroo.core.rewriter.passes.structure import StructuralTransformer
  from ml_switcheroo.semantics.manager import SemanticsManager

  ctx = RewriterContext(semantics=SemanticsManager(), config=RuntimeConfig())

  class SuperShim(cst.CSTTransformer):
    """Stub `cst.CSTTransformer` subclass implementing `leave_Attribute` for structure tests."""

    def leave_Attribute(self, original: cst.Attribute, updated: cst.Attribute) -> cst.BaseExpression:
      """Execute `updated.with_changes` to support leave attribute in structure tests.

      Args:
          original: Input `original` parameter value (`cst.Attribute`).
          updated: Input `updated` parameter value (`cst.Attribute`).

      Returns:
          Resulting `cst.BaseExpression` from leave Attribute.
      """
      return updated.with_changes(value=cst.Name("shimmed"))

  class ShimmedStructuralPass(StructuralTransformer, SuperShim):
    """Minimal `StructuralTransformer` test double used in structure tests."""

  transformer = ShimmedStructuralPass(ctx)
  mod = cst.parse_module("a.b")
  stmt = cst.ensure_type(mod.body[0], cst.SimpleStatementLine)
  expr_stmt = cst.ensure_type(stmt.body[0], cst.Expr)
  attr = cst.ensure_type(expr_stmt.value, cst.Attribute)

  # Needs to bypass our custom leave_Attribute logic
  attr_updated = cst.ensure_type(transformer.leave_Attribute(attr, attr), cst.Attribute)
  assert cst.ensure_type(attr_updated.value, cst.Name).value == "shimmed"


class DummySemanticsExtra(SemanticsManager):
  """Test double `DummySemanticsExtra` used in structure tests."""

  def __init__(self) -> None:
    """Initializes the DummySemantics instance."""
    super().__init__(load_defaults=False)
    self.definitions: dict[str, tuple[str, dict[str, JsonValue]]] = {}
    self.configs: dict[str, dict[str, JsonValue]] = {}
    self.framework_configs = self.configs
    self.variants: dict[tuple[str, str], dict[str, JsonValue]] = {}
    self.known_magic_args: set[str] = set()

  def resolve_definition(self, *args: JsonValue, **kwargs: JsonValue) -> None:
    """Mock implementation of resolve definition.

    Args:
        args: Positional argument specifications (`JsonValue`).
        kwargs: Keyword argument bindings (`JsonValue`).
    """
    return

  def get_standard_module(self, *args: JsonValue, **kwargs: JsonValue) -> str:
    """Mock implementation of get standard module.

    Args:
        args: Positional argument specifications (`JsonValue`).
        kwargs: Keyword argument bindings (`JsonValue`).

    Returns:
        Resulting `str` from get standard module.
    """
    return "nn"

  def resolve_variant(self, op_id: str, fw: str) -> dict[str, JsonValue] | None:
    """Mock implementation of resolve variant.

    Args:
        op_id: Input `op_id` parameter value (`str`).
        fw: Framework identifier string (e.g., torch, jax, mlx).

    Returns:
        Resulting `dict[str, JsonValue] | None` from resolve variant.
    """
    _ret_v: dict[str, JsonValue] | None = self.variants.get((op_id, fw))
    return _ret_v

  def get_framework_config(self, framework: str) -> dict[str, JsonValue]:
    """Mock implementation of get framework configuration.

    Args:
        framework: Framework identifier string.

    Returns:
        Resulting `dict[str, JsonValue]` from get framework config.
    """
    _ret_v: dict[str, JsonValue] = self.configs.get(framework, {})
    return _ret_v

  def get_definition(self, api: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Mock implementation of get definition.

    Args:
        api: Input `api` parameter value (`str`).

    Returns:
        Resulting `tuple[str, dict[str, JsonValue]] | None` from get definition.
    """
    _ret_v: tuple[str, dict[str, JsonValue]] | None = self.definitions.get(api)
    return _ret_v


def get_transformer_extra() -> tuple[StructuralTransformer, DummySemanticsExtra, RewriterContext]:
  """Gets transformer.

  Returns:
      Resulting `tuple[StructuralTransformer, DummySemanticsExtra, RewriterContext]` from get transformer extra.
  """
  sem = DummySemanticsExtra()
  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  ctx = RewriterContext(sem, cfg)
  return (StructuralTransformer(ctx), sem, ctx)


def test_preamble_and_docstring() -> None:
  """Verify `get_transformer` emits `print(1)` for preamble and docstring."""
  (transformer, _sem, ctx) = get_transformer()
  func = cst.ensure_type(cst.parse_module('def __init__(self):\n  """doc"""\n  pass').body[0], cst.FunctionDef)
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
  func_simple = cst.ensure_type(cst.parse_module("def foo(): print(2)").body[0], cst.FunctionDef)
  res_simple = transformer._convert_to_indented_block(func_simple)
  assert isinstance(res_simple.body, cst.IndentedBlock)


def test_strip_argument_from_signature() -> None:
  """Verify `get_transformer` handles strip argument from signature as expected."""
  (transformer, _sem, _ctx) = get_transformer()
  func = cst.ensure_type(cst.parse_module("def f(x, y): pass").body[0], cst.FunctionDef)
  res = transformer._strip_argument_from_signature(func, "x")
  assert res.params.params[0].name.value == "y"


def test_fix_comma() -> None:
  """Fixes comma."""
  (transformer, _sem, _ctx) = get_transformer()
  func = cst.ensure_type(cst.parse_module("def f(x, y): pass").body[0], cst.FunctionDef)
  params = list(func.params.params)
  params[-1] = params[-1].with_changes(comma=cst.Comma())
  res = transformer._fix_comma(func, params)
  assert res.params.params[-1].comma == cst.MaybeSentinel.DEFAULT


def test_leave_module_preamble_empty_stmts() -> None:
  """Verify `get_transformer` handles leave module preamble empty stmts as expected."""
  (transformer, _sem, ctx) = get_transformer()
  mod = cst.parse_module("a = 1")
  ctx.module_preamble.append("invalid code ###")
  new_mod = transformer.leave_Module(mod, mod)
  assert "invalid code" not in new_mod.code
  assert new_mod is mod


def test_leave_name_in_annotation_success() -> None:
  """Verify `get_transformer` handles leave name in annotation success as expected."""
  (transformer, sem, _ctx) = get_transformer()
  transformer._in_annotation = True
  sem.definitions["UnknownType"] = ("UnknownType", {})
  sem.variants[("UnknownType", "jax")] = {"api": "jnp.MappedType"}
  name = cst.Name("UnknownType")
  res = transformer.leave_Name(name, name)
  assert transformer._cst_to_string(res) == "jnp.MappedType"


def test_leave_attribute_in_annotation_success() -> None:
  """Verify `get_transformer` handles leave attribute in annotation success as expected."""
  (transformer, sem, _ctx) = get_transformer()
  transformer._in_annotation = True
  sem.definitions["Unknown.Type"] = ("UnknownType", {})
  sem.variants[("UnknownType", "jax")] = {"api": "jnp.MappedType"}
  attr = cst.Attribute(value=cst.Name("Unknown"), attr=cst.Name("Type"))
  res = transformer.leave_Attribute(attr, attr)
  assert transformer._cst_to_string(res) == "jnp.MappedType"


def test_leave_classdef_raw_name_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `get_transformer` handles leave classdef raw name fallback as expected.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  (transformer, sem, _ctx) = get_transformer()
  sem.configs["torch"] = {"traits": {"module_base": "torch.nn.Module"}}
  sem.configs["jax"] = {"traits": {"module_base": "flax.nnx.Module"}}
  class_node = cst.ensure_type(cst.parse_module("class Net(nn.Module): pass").body[0], cst.ClassDef)
  transformer.visit_ClassDef(class_node)
  orig_gqn = transformer._get_qualified_name
  monkeypatch.setattr(transformer, "_get_qualified_name", lambda n: None, raising=False)
  new_node = cst.ensure_type(transformer.leave_ClassDef(class_node, class_node), cst.ClassDef)
  monkeypatch.setattr(transformer, "_get_qualified_name", orig_gqn, raising=False)
  base0 = transformer._cst_to_string(new_node.bases[0].value)
  assert base0 is not None
  assert "flax.nnx.Module" in base0


def test_convert_to_indented_block_fallback() -> None:
  """Converts to indented block fallback."""
  (transformer, _sem, _ctx) = get_transformer()
  func = cst.ensure_type(cst.parse_module("def foo():\n  pass").body[0], cst.FunctionDef)
  assert transformer._convert_to_indented_block(func) is func


def test_ensure_super_init_already_has() -> None:
  """Verify `get_transformer` handles ensure super init already has as expected."""
  (transformer, _sem, _ctx) = get_transformer()
  func = cst.ensure_type(cst.parse_module("def __init__(self):\n  super().__init__()").body[0], cst.FunctionDef)
  assert transformer._ensure_super_init(func) is func


def test_strip_super_init_no_body() -> None:
  """Verify `get_transformer` handles strip super init no body as expected."""
  (transformer, _sem, _ctx) = get_transformer()
  func = cst.ensure_type(cst.parse_module("def f(): pass").body[0], cst.FunctionDef)
  func = func.with_changes(body=cst.Pass())
  assert transformer._strip_super_init(func) is func


def test_has_super_init_false() -> None:
  """Checks if has super initialization false."""
  (transformer, _sem, _ctx) = get_transformer()
  func = cst.ensure_type(cst.parse_module("def __init__(self):\n  pass").body[0], cst.FunctionDef)
  assert transformer._has_super_init(func) is False


def test_update_docstring_fallback() -> None:
  """Verify `get_transformer` handles update docstring fallback as expected."""
  (transformer, _sem, _ctx) = get_transformer()
  func0 = cst.ensure_type(cst.parse_module("def f(): pass").body[0], cst.FunctionDef)
  func0 = func0.with_changes(body=cst.SimpleStatementSuite(body=[]))
  assert transformer._update_docstring(func0, [("a", "b")]) is func0
  func = cst.ensure_type(cst.parse_module("def f():\n  x = 1").body[0], cst.FunctionDef)
  assert transformer._update_docstring(func, [("a", "b")]) is func
  func2 = cst.ensure_type(cst.parse_module("def f():\n  'doc'").body[0], cst.FunctionDef)
  assert transformer._update_docstring(func2, [("a", "b")]) is func2


def test_visit_classdef_raw_name_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `get_transformer` handles visit classdef raw name fallback as expected.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  (transformer, sem, ctx) = get_transformer()
  sem.configs["torch"] = {"traits": {"module_base": "torch.nn.Module"}}
  class_node = cst.ensure_type(cst.parse_module("class Net(nn.Module): pass").body[0], cst.ClassDef)
  orig_gqn = transformer._get_qualified_name
  monkeypatch.setattr(transformer, "_get_qualified_name", lambda n: None, raising=False)
  transformer.visit_ClassDef(class_node)
  monkeypatch.setattr(transformer, "_get_qualified_name", orig_gqn, raising=False)
  assert ctx.in_module_class is True


def test_leave_attribute_fallback_super() -> None:
  """Verify `get_transformer` handles leave attribute fallback super as expected."""
  (transformer, _sem, _ctx) = get_transformer()
  transformer._in_annotation = False
  attr = cst.Attribute(value=cst.Name("Unknown"), attr=cst.Name("Type"))
  res = transformer.leave_Attribute(attr, attr)
  assert res is attr
