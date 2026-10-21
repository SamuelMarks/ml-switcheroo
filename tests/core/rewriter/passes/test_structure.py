"""Test suite for the Structure module."""

import typing
from pydantic import JsonValue

import libcst as cst
import pytest
from ml_switcheroo_ir.schema.ghost import SemanticTier

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.core.rewriter.passes.structure import StructuralPass, StructuralTransformer
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.semantics.schema import StructuralTraits


class MockSemantics(SemanticsManager):
  """Test helper class MockSemantics."""

  def __init__(self) -> None:
    """Initializes the MockSemantics instance."""
    self.data = {}
    setattr(
      getattr({"v": self}, "__getitem__")("v"),
      "framework_configs",
      getattr(
        {
          "v": {
            "torch": {"traits": {"module_base": "torch.nn.Module", "forward_method": "forward"}},
            "jax": getattr(
              {
                "v": {
                  "traits": getattr(
                    {
                      "v": {
                        "module_base": "flax.nnx.Module",
                        "forward_method": "__call__",
                        "init_method_name": "__init__",
                        "inject_magic_args": getattr(
                          {
                            "v": [
                              getattr(
                                {
                                  "v": (
                                    getattr({"v": "rngs"}, "__getitem__")("v"),
                                    getattr({"v": "nnx.Rngs"}, "__getitem__")("v"),
                                  )
                                },
                                "__getitem__",
                              )("v")
                            ]
                          },
                          "__getitem__",
                        )("v"),
                        "requires_super_init": False,
                      }
                    },
                    "__getitem__",
                  )("v"),
                  "alias": {"module": "flax.nnx", "name": "nnx"},
                }
              },
              "__getitem__",
            )("v"),
          }
        },
        "__getitem__",
      )("v"),
    )
    self.data["Tensor"] = {"variants": {"jax": {"api": "jax.Array"}}}
    self._reverse_index = {"torch.Tensor": ("Tensor", self.data["Tensor"])}

  def get_framework_config(self, fw: str) -> dict[str, JsonValue]:
    """Mock implementation of get framework configuration.

    Args:
        fw: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self.framework_configs.get(fw, {})

  def get_definition(self, name: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Mock implementation of get definition.

    Args:
        name: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self._reverse_index.get(name)

  def resolve_variant(self, aid: str, fw: str) -> dict[str, JsonValue] | None:
    """Mock implementation of resolve variant.

    Args:
        aid: Test argument or fixture.
        fw: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    if getattr(
      {
        "v": getattr({"v": aid}, "__getitem__")("v")
        in getattr({"v": getattr({"v": self}, "__getitem__")("v").data}, "__getitem__")("v")
        and getattr({"v": fw}, "__getitem__")("v")
        in getattr(
          {
            "v": getattr(
              {
                "v": getattr(
                  {"v": getattr({"v": getattr({"v": self}, "__getitem__")("v").data}, "__getitem__")("v")[aid]},
                  "__getitem__",
                )("v").get
              },
              "__getitem__",
            )("v")(getattr({"v": "variants"}, "__getitem__")("v"), getattr({"v": {}}, "__getitem__")("v"))
          },
          "__getitem__",
        )("v")
      },
      "__getitem__",
    )("v"):
      _ret_v: dict[str, JsonValue] | None = getattr(
        {
          "v": getattr(
            {
              "v": getattr(
                {"v": getattr({"v": getattr({"v": self}, "__getitem__")("v").data}, "__getitem__")("v")[aid]},
                "__getitem__",
              )("v")["variants"]
            },
            "__getitem__",
          )("v")[fw]
        },
        "__getitem__",
      )("v")
      return _ret_v
    return None


@pytest.fixture
def run_pass() -> typing.Callable[[str], str]:
  """Execute run pass helper logic.

  Returns:
      Computed test result or fixture value.
  """
  semantics = MockSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  context = RewriterContext(semantics, config)
  context.alias_map["torch"] = "torch"
  context.alias_map["torch.nn"] = "torch.nn"

  def _transform(code: str) -> str:
    """Helper to  transform.

    Args:
        code: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    module = cst.parse_module(code)
    struct_pass = StructuralPass()
    return struct_pass.transform(module, context).code

  return _transform


def test_class_base_rewrite(run_pass: typing.Callable[[str], str]) -> None:
  """Verifies the behavior of class base rewrite.

  Args:
      run_pass: Test argument or fixture.
  """
  code: str = "class Net(torch.nn.Module): pass"
  res: str = run_pass(code)
  assert "class Net(flax.nnx.Module):" in res


def test_class_base_rewrite_aliased(run_pass: typing.Callable[[str], str]) -> None:
  """Verifies the behavior of class base rewrite aliased.

  Args:
      run_pass: Test argument or fixture.
  """
  code: str = "class Net(torch.nn.Module): pass"
  res: str = run_pass(code)
  assert "flax.nnx.Module" in res


def test_method_renaming(run_pass: typing.Callable[[str], str]) -> None:
  """Verifies the behavior of method renaming.

  Args:
      run_pass: Test argument or fixture.
  """
  code: str = "\nclass Net(torch.nn.Module):\n    def forward(self, x): pass\n"
  res: str = run_pass(code)
  assert "def __call__(self, x):" in res


def test_magic_arg_injection(run_pass: typing.Callable[[str], str]) -> None:
  """Verifies the behavior of magic argument injection.

  Args:
      run_pass: Test argument or fixture.
  """
  code: str = "\nclass Net(torch.nn.Module):\n    def __init__(self, dim): pass\n"
  res: str = run_pass(code)
  assert "def __init__(self, rngs: nnx.Rngs, dim):" in res


def test_super_init_stripping(run_pass: typing.Callable[[str], str]) -> None:
  """Verifies the behavior of super initialization stripping.

  Args:
      run_pass: Test argument or fixture.
  """
  code: str = "\nclass Net(torch.nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.x = 1\n"
  res: str = run_pass(code)
  assert "super().__init__()" not in res
  assert "self.x = 1" in res


def test_type_hint_rewrite(run_pass: typing.Callable[[str], str]) -> None:
  """Verifies the behavior of type hint rewrite.

  Args:
      run_pass: Test argument or fixture.
  """
  code: str = "def f(x: torch.Tensor): pass"
  res: str = run_pass(code)
  assert "x: jax.Array" in res


def test_ignore_non_module_classes(run_pass: typing.Callable[[str], str]) -> None:
  """Verifies the behavior of ignore non module classes.

  Args:
      run_pass: Test argument or fixture.
  """
  code: str = "\nclass Data:\n    def forward(self): pass\n"
  res: str = run_pass(code)
  assert "class Data:" in res
  assert "def forward(self):" in res


# --- Merged from test_structure_extra.py ---


class DummySemantics(SemanticsManager):
  """Test helper class DummySemantics."""

  def __init__(self) -> None:
    """Initializes the DummySemantics instance."""
    self.configs: dict[str, typing.Hashable | typing.Sized] = {}
    setattr(
      getattr({"v": self}, "__getitem__")("v"),
      "framework_configs",
      getattr({"v": getattr({"v": self}, "__getitem__")("v").configs}, "__getitem__")("v"),
    )
    self.definitions: dict[str, typing.Hashable | typing.Sized] = {}
    self.variants: dict[str, typing.Hashable | typing.Sized] = {}
    self.verified = True
    self.known_magic_args: set[str] = set()

  def get_framework_config(self, fw: str) -> dict[str, JsonValue]:
    """Mock implementation of get framework configuration.

    Args:
        fw: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    _ret_v: dict[str, JsonValue] = getattr(
      {
        "v": getattr(
          {"v": getattr({"v": getattr({"v": self}, "__getitem__")("v").configs}, "__getitem__")("v").get}, "__getitem__"
        )("v")(getattr({"v": fw}, "__getitem__")("v"), getattr({"v": {}}, "__getitem__")("v"))
      },
      "__getitem__",
    )("v")
    return _ret_v

  def get_definition(self, name: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Mock implementation of get definition.

    Args:
        name: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    _ret_v: tuple[str, dict[str, JsonValue]] | None = getattr(
      {
        "v": getattr(
          {"v": getattr({"v": getattr({"v": self}, "__getitem__")("v").definitions}, "__getitem__")("v").get},
          "__getitem__",
        )("v")(getattr({"v": name}, "__getitem__")("v"))
      },
      "__getitem__",
    )("v")
    return _ret_v

  def resolve_variant(self, abstract_id: str, fw: str) -> dict[str, JsonValue] | None:
    """Mock implementation of resolve variant.

    Args:
        abstract_id: Test argument or fixture.
        fw: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    _ret_v: dict[str, JsonValue] | None = getattr(
      {
        "v": getattr(
          {"v": getattr({"v": getattr({"v": self}, "__getitem__")("v").variants}, "__getitem__")("v").get}, "__getitem__"
        )("v")(
          getattr(
            {"v": (getattr({"v": abstract_id}, "__getitem__")("v"), getattr({"v": fw}, "__getitem__")("v"))},
            "__getitem__",
          )("v")
        )
      },
      "__getitem__",
    )("v")
    return _ret_v


def get_transformer() -> tuple[StructuralTransformer, DummySemantics, RewriterContext]:
  """Gets transformer.

  Returns:
      Computed test result or fixture value.
  """
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
  assert getattr(
    {
      "v": getattr({"v": getattr({"v": transformer}, "__getitem__")("v")._is_framework_base}, "__getitem__")("v")(
        getattr({"v": None}, "__getitem__")("v")
      )
    },
    "__getitem__",
  )("v") is getattr({"v": False}, "__getitem__")("v")


def test_is_framework_base_traits_object() -> None:
  """Checks if is framework base traits object."""
  (transformer, sem, ctx) = get_transformer()

  class DummyTraits:
    """Test helper class DummyTraits."""

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
  new_name = transformer.leave_Name(name, name)
  assert new_name is name


def test_leave_attribute_not_in_annotation() -> None:
  """Verifies the behavior of leave attribute not in annotation."""
  (transformer, sem, ctx) = get_transformer()
  attr = cst.Attribute(value=cst.Name("x"), attr=cst.Name("y"))
  new_attr = transformer.leave_Attribute(attr, attr)
  assert new_attr is attr


def test_visit_classdef_fallback_and_error() -> None:
  """Verifies the behavior of visit classdef fallback and correctly handling an error."""
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
  """Verifies the behavior of leave classdef unmapped base."""
  (transformer, sem, ctx) = get_transformer()
  sem.configs["torch"] = {"traits": {"module_base": "torch.nn.Module"}}
  sem.configs["jax"] = {"traits": {"module_base": "flax.nnx.Module"}}
  class_node = cst.ensure_type(cst.parse_module("class Net(nn.Module, Other): pass").body[0], cst.ClassDef)
  transformer.visit_ClassDef(class_node)
  new_node = transformer.leave_ClassDef(class_node, class_node)
  assert getattr({"v": "flax.nnx.Module"}, "__getitem__")("v") in getattr(
    {
      "v": getattr({"v": getattr({"v": transformer}, "__getitem__")("v")._cst_to_string}, "__getitem__")("v")(
        getattr(
          {
            "v": getattr(
              {"v": getattr({"v": getattr({"v": new_node}, "__getitem__")("v").bases}, "__getitem__")("v")[0]},
              "__getitem__",
            )("v").value
          },
          "__getitem__",
        )("v")
      )
    },
    "__getitem__",
  )("v")
  assert getattr({"v": "Other"}, "__getitem__")("v") in getattr(
    {
      "v": getattr({"v": getattr({"v": transformer}, "__getitem__")("v")._cst_to_string}, "__getitem__")("v")(
        getattr(
          {
            "v": getattr(
              {"v": getattr({"v": getattr({"v": new_node}, "__getitem__")("v").bases}, "__getitem__")("v")[1]},
              "__getitem__",
            )("v").value
          },
          "__getitem__",
        )("v")
      )
    },
    "__getitem__",
  )("v")


def test_leave_functiondef_no_stack() -> None:
  """Verifies the behavior of leave functiondef no stack."""
  (transformer, sem, ctx) = get_transformer()
  func = cst.ensure_type(cst.parse_module("def foo(): pass").body[0], cst.FunctionDef)
  assert transformer.leave_FunctionDef(func, func) is func


def test_leave_functiondef_renaming() -> None:
  """Verifies the behavior of leave functiondef renaming."""
  (transformer, sem, ctx) = get_transformer()
  sem.configs["jax"] = {"traits": {"init_method_name": "setup"}}
  func = cst.ensure_type(cst.parse_module("def __init__(self): pass").body[0], cst.FunctionDef)
  transformer.visit_FunctionDef(func)
  ctx.in_module_class = True
  ctx.signature_stack[-1].is_module_method = True
  new_func = transformer.leave_FunctionDef(func, func)
  assert new_func.name.value == "setup"


def test_leave_functiondef_magic_args() -> None:
  """Verifies the behavior of leave functiondef magic arguments."""
  (transformer, sem, ctx) = get_transformer()
  sem.known_magic_args.add("rngs")
  sem.configs["jax"] = {
    "traits": {"auto_strip_magic_args": True, "strip_magic_args": ["ctx"], "inject_magic_args": [("rngs", "int")]}
  }
  func = cst.ensure_type(cst.parse_module("def __init__(self, ctx, rngs): pass").body[0], cst.FunctionDef)
  transformer.visit_FunctionDef(func)
  ctx.in_module_class = True
  ctx.signature_stack[-1].is_module_method = True
  new_func = transformer.leave_FunctionDef(func, func)
  params: list[str] = getattr(
    {
      "v": [
        getattr({"v": getattr({"v": p}, "__getitem__")("v").name}, "__getitem__")("v").value
        for p in getattr({"v": getattr({"v": new_func}, "__getitem__")("v").params}, "__getitem__")("v").params
        if isinstance(getattr({"v": getattr({"v": p}, "__getitem__")("v").name}, "__getitem__")("v"), cst.Name)
      ]
    },
    "__getitem__",
  )("v")
  assert "ctx" not in params
  assert "rngs" in params


def test_super_init_logic() -> None:
  """Verifies the behavior of super initialization logic."""
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
  """Verify leave module preamble no cover behavior."""
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
  """Verify leave attribute super shim behavior."""
  from ml_switcheroo.config import RuntimeConfig
  from ml_switcheroo.core.rewriter.context import RewriterContext
  from ml_switcheroo.core.rewriter.passes.structure import StructuralTransformer
  from ml_switcheroo.semantics.manager import SemanticsManager

  ctx = RewriterContext(semantics=SemanticsManager(), config=RuntimeConfig())

  class SuperShim(cst.CSTTransformer):
    """Test helper class SuperShim."""

    def leave_Attribute(self, original: cst.Attribute, updated: cst.Attribute) -> cst.BaseExpression:
      """Execute leave Attribute helper logic.

      Args:
          original: Test argument or fixture.
          updated: Test argument or fixture.

      Returns:
          Computed test result or fixture value.
      """
      return updated.with_changes(value=cst.Name("shimmed"))

  class ShimmedStructuralPass(StructuralTransformer, SuperShim):
    """Test helper class ShimmedStructuralPass."""

    pass

  transformer = ShimmedStructuralPass(ctx)
  mod = cst.parse_module("a.b")
  attr = getattr(getattr(mod.body[0], "body")[0], "value")

  # Needs to bypass our custom leave_Attribute logic
  attr_updated = transformer.leave_Attribute(attr, attr)
  assert getattr(
    {
      "v": getattr(
        {
          "v": getattr({"v": getattr({"v": cst}, "__getitem__")("v").ensure_type}, "__getitem__")("v")(
            getattr({"v": getattr({"v": attr_updated}, "__getitem__")("v").value}, "__getitem__")("v"),
            getattr({"v": getattr({"v": cst}, "__getitem__")("v").Name}, "__getitem__")("v"),
          )
        },
        "__getitem__",
      )("v").value
    },
    "__getitem__",
  )("v") == getattr({"v": "shimmed"}, "__getitem__")("v")


# --- Merged from test_structure_extra2.py ---


class DummySemanticsExtra:
  """Test helper class DummySemanticsExtra."""

  def __init__(self) -> None:
    """Initializes the DummySemantics instance."""
    self.definitions: dict[str, typing.Hashable | typing.Sized] = {}
    self.configs: dict[str, typing.Hashable | typing.Sized] = {}
    self.framework_configs = self.configs
    self.variants: dict[str, typing.Hashable | typing.Sized] = {}
    self.known_magic_args: set[str] = set()

  def resolve_definition(self, *args: JsonValue, **kwargs: JsonValue) -> None:
    """Mock implementation of resolve definition.

    Args:
        args: Test argument or fixture.
        kwargs: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return None

  def get_standard_module(self, *args: JsonValue, **kwargs: JsonValue) -> str:
    """Mock implementation of get standard module.

    Args:
        args: Test argument or fixture.
        kwargs: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return "nn"

  def resolve_variant(self, op_id: str, fw: str) -> dict[str, JsonValue] | None:
    """Mock implementation of resolve variant.

    Args:
        op_id: Test argument or fixture.
        fw: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    _ret_v: dict[str, JsonValue] | None = getattr(
      {
        "v": getattr(
          {"v": getattr({"v": getattr({"v": self}, "__getitem__")("v").variants}, "__getitem__")("v").get}, "__getitem__"
        )("v")(
          getattr(
            {"v": (getattr({"v": op_id}, "__getitem__")("v"), getattr({"v": fw}, "__getitem__")("v"))}, "__getitem__"
          )("v")
        )
      },
      "__getitem__",
    )("v")
    return _ret_v

  def get_framework_config(self, framework: str) -> dict[str, JsonValue]:
    """Mock implementation of get framework configuration.

    Returns:
        Computed test result or fixture value.
    """
    _ret_v: dict[str, JsonValue] = getattr(
      {
        "v": getattr(
          {"v": getattr({"v": getattr({"v": self}, "__getitem__")("v").configs}, "__getitem__")("v").get}, "__getitem__"
        )("v")(getattr({"v": framework}, "__getitem__")("v"), getattr({"v": {}}, "__getitem__")("v"))
      },
      "__getitem__",
    )("v")
    return _ret_v

  def get_definition(self, api: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Mock implementation of get definition.

    Args:
        api: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    _ret_v: tuple[str, dict[str, JsonValue]] | None = getattr(
      {
        "v": getattr(
          {"v": getattr({"v": getattr({"v": self}, "__getitem__")("v").definitions}, "__getitem__")("v").get},
          "__getitem__",
        )("v")(getattr({"v": api}, "__getitem__")("v"))
      },
      "__getitem__",
    )("v")
    return _ret_v


def get_transformer_extra() -> tuple[StructuralTransformer, DummySemanticsExtra, RewriterContext]:
  """Gets transformer.

  Returns:
      Computed test result or fixture value.
  """
  sem = DummySemanticsExtra()
  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  ctx = RewriterContext(sem, cfg)
  return (StructuralTransformer(ctx), sem, ctx)


def test_preamble_and_docstring() -> None:
  """Verify preamble and docstring behavior."""
  (transformer, sem, ctx) = get_transformer()
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
  """Verifies the behavior of strip argument from signature."""
  (transformer, sem, ctx) = get_transformer()
  func = cst.ensure_type(cst.parse_module("def f(x, y): pass").body[0], cst.FunctionDef)
  res = transformer._strip_argument_from_signature(func, "x")
  assert res.params.params[0].name.value == "y"


def test_fix_comma() -> None:
  """Fixes comma."""
  (transformer, sem, ctx) = get_transformer()
  func = cst.ensure_type(cst.parse_module("def f(x, y): pass").body[0], cst.FunctionDef)
  params = list(func.params.params)
  params[-1] = params[-1].with_changes(comma=cst.Comma())
  res = transformer._fix_comma(func, params)
  assert res.params.params[-1].comma == cst.MaybeSentinel.DEFAULT


def test_leave_module_preamble_empty_stmts() -> None:
  """Verifies the behavior of leave module preamble empty stmts."""
  (transformer, sem, ctx) = get_transformer()
  mod = cst.parse_module("a = 1")
  ctx.module_preamble.append("invalid code ###")
  new_mod = transformer.leave_Module(mod, mod)
  assert "invalid code" not in new_mod.code
  assert new_mod is mod


def test_leave_name_in_annotation_success() -> None:
  """Verifies the behavior of leave name in annotation successfully."""
  (transformer, sem, ctx) = get_transformer()
  transformer._in_annotation = True
  sem.definitions["UnknownType"] = ("UnknownType", {})
  getattr(getattr({"v": getattr({"v": sem}, "__getitem__")("v").variants}, "__getitem__")("v"), "__setitem__")(
    (getattr({"v": "UnknownType"}, "__getitem__")("v"), getattr({"v": "jax"}, "__getitem__")("v")),
    getattr({"v": {"api": getattr({"v": "jnp.MappedType"}, "__getitem__")("v")}}, "__getitem__")("v"),
  )
  name = cst.Name("UnknownType")
  res = transformer.leave_Name(name, name)
  assert transformer._cst_to_string(res) == "jnp.MappedType"


def test_leave_attribute_in_annotation_success() -> None:
  """Verifies the behavior of leave attribute in annotation successfully."""
  (transformer, sem, ctx) = get_transformer()
  transformer._in_annotation = True
  sem.definitions["Unknown.Type"] = ("UnknownType", {})
  getattr(getattr({"v": getattr({"v": sem}, "__getitem__")("v").variants}, "__getitem__")("v"), "__setitem__")(
    (getattr({"v": "UnknownType"}, "__getitem__")("v"), getattr({"v": "jax"}, "__getitem__")("v")),
    getattr({"v": {"api": getattr({"v": "jnp.MappedType"}, "__getitem__")("v")}}, "__getitem__")("v"),
  )
  attr = cst.Attribute(value=cst.Name("Unknown"), attr=cst.Name("Type"))
  res = transformer.leave_Attribute(attr, attr)
  assert transformer._cst_to_string(res) == "jnp.MappedType"


def test_leave_classdef_raw_name_fallback() -> None:
  """Verifies the behavior of leave classdef raw name fallback."""
  (transformer, sem, ctx) = get_transformer()
  sem.configs["torch"] = {"traits": {"module_base": "torch.nn.Module"}}
  sem.configs["jax"] = {"traits": {"module_base": "flax.nnx.Module"}}
  class_node = cst.ensure_type(cst.parse_module("class Net(nn.Module): pass").body[0], cst.ClassDef)
  transformer.visit_ClassDef(class_node)
  orig_gqn = transformer._get_qualified_name
  setattr(transformer, "_get_qualified_name", lambda n: None)
  new_node = transformer.leave_ClassDef(class_node, class_node)
  setattr(transformer, "_get_qualified_name", orig_gqn)
  assert getattr({"v": "flax.nnx.Module"}, "__getitem__")("v") in getattr(
    {
      "v": getattr({"v": getattr({"v": transformer}, "__getitem__")("v")._cst_to_string}, "__getitem__")("v")(
        getattr(
          {
            "v": getattr(
              {"v": getattr({"v": getattr({"v": new_node}, "__getitem__")("v").bases}, "__getitem__")("v")[0]},
              "__getitem__",
            )("v").value
          },
          "__getitem__",
        )("v")
      )
    },
    "__getitem__",
  )("v")


def test_convert_to_indented_block_fallback() -> None:
  """Converts to indented block fallback."""
  (transformer, sem, ctx) = get_transformer()
  func = cst.ensure_type(cst.parse_module("def foo():\n  pass").body[0], cst.FunctionDef)
  assert transformer._convert_to_indented_block(func) is func


def test_ensure_super_init_already_has() -> None:
  """Verifies the behavior of ensure super initialization already has."""
  (transformer, sem, ctx) = get_transformer()
  func = cst.ensure_type(cst.parse_module("def __init__(self):\n  super().__init__()").body[0], cst.FunctionDef)
  assert transformer._ensure_super_init(func) is func


def test_strip_super_init_no_body() -> None:
  """Verifies the behavior of strip super initialization no body."""
  (transformer, sem, ctx) = get_transformer()
  func = cst.ensure_type(cst.parse_module("def f(): pass").body[0], cst.FunctionDef)
  func = func.with_changes(body=cst.Pass())
  assert transformer._strip_super_init(func) is func


def test_has_super_init_false() -> None:
  """Checks if has super initialization false."""
  (transformer, sem, ctx) = get_transformer()
  func = cst.ensure_type(cst.parse_module("def __init__(self):\n  pass").body[0], cst.FunctionDef)
  assert transformer._has_super_init(func) is False


def test_update_docstring_fallback() -> None:
  """Verify update docstring fallback behavior."""
  (transformer, sem, ctx) = get_transformer()
  func0 = cst.ensure_type(cst.parse_module("def f(): pass").body[0], cst.FunctionDef)
  func0 = func0.with_changes(body=cst.SimpleStatementSuite(body=[]))
  assert transformer._update_docstring(func0, [("a", "b")]) is func0
  func = cst.ensure_type(cst.parse_module("def f():\n  x = 1").body[0], cst.FunctionDef)
  assert transformer._update_docstring(func, [("a", "b")]) is func
  func2 = cst.ensure_type(cst.parse_module("def f():\n  'doc'").body[0], cst.FunctionDef)
  assert transformer._update_docstring(func2, [("a", "b")]) is func2


def test_visit_classdef_raw_name_fallback() -> None:
  """Verifies the behavior of visit classdef raw name fallback."""
  (transformer, sem, ctx) = get_transformer()
  sem.configs["torch"] = {"traits": {"module_base": "torch.nn.Module"}}
  class_node = cst.ensure_type(cst.parse_module("class Net(nn.Module): pass").body[0], cst.ClassDef)
  orig_gqn = transformer._get_qualified_name
  setattr(transformer, "_get_qualified_name", lambda n: None)
  transformer.visit_ClassDef(class_node)
  setattr(transformer, "_get_qualified_name", orig_gqn)
  assert ctx.in_module_class


def test_leave_attribute_fallback_super() -> None:
  """Verifies the behavior of leave attribute fallback super."""
  (transformer, sem, ctx) = get_transformer()
  transformer._in_annotation = False
  attr = cst.Attribute(value=cst.Name("Unknown"), attr=cst.Name("Type"))
  res = transformer.leave_Attribute(attr, attr)
  assert res is attr
