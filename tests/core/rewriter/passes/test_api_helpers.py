"""Test suite for the Api Helpers module."""

import pytest
import typing
import libcst as cst
from ml_switcheroo.core.rewriter.passes.api_helpers import ApiHelpersMixin


class MockTracer:
  """Mock Tracer class for testing purposes."""

  def log_match(self, *args: typing.Any, **kwargs: typing.Any) -> None:
    """Mock implementation of log match."""
    pass


class MockSemantics:
  """Mock Semantics class for testing purposes."""

  def __init__(self, defs: dict[str, tuple[str, dict[str, typing.Any]]], configs: dict[str, typing.Any]) -> None:
    """Initializes the MockSemantics instance."""
    self.defs = defs
    self.framework_configs = configs

  def get_definition(self, name: str) -> typing.Optional[tuple[str, dict[str, typing.Any]]]:
    """Mock implementation of get definition."""
    return self.defs.get(name)

  def is_verified(self, name: str) -> bool:
    """Mock implementation of is verified."""
    return self.defs.get(name, ("", {"verified": True}))[1].get("verified", True)

  def resolve_variant(self, abstract_id: str, target: str) -> typing.Optional[dict[str, typing.Any]]:
    """Mock implementation of resolve variant."""
    return self.defs.get(abstract_id, ("", {}))[1].get("variants", {}).get(target)

  def get_framework_config(self, target: str) -> dict[str, typing.Any]:
    """Mock implementation of get framework configuration."""
    return self.framework_configs.get(target, {})


class MockHookContext:
  """Mock Hook Context class for testing purposes."""

  def __init__(self) -> None:
    """Initializes the MockHookContext instance."""
    self.stmts: list[typing.Any] = []

  def inject_preamble(self, s: typing.Any) -> None:
    """Mock implementation of inject preamble."""
    self.stmts.append(s)


class MockContext:
  """Mock Context class for testing purposes."""

  def __init__(self) -> None:
    """Initializes the MockContext instance."""
    self.alias_map = {"np": "numpy"}
    self.hook_context = MockHookContext()


class MockConfig:
  """Mock Config class for testing purposes."""

  def __init__(self) -> None:
    """Initializes the MockConfig instance."""
    self.source_framework = "torch"
    self.target_framework = "jax"
    self.source_flavour = "torch.nn"


class MockHelper(ApiHelpersMixin):
  """Mock Helper class for testing purposes."""

  def __init__(self) -> None:
    """Initializes the MockHelper instance."""
    self.context = MockContext()
    self.semantics = MockSemantics(
      {
        "numpy.add": ("numpy.add", {"verified": True, "variants": {"jax": {"api": "jnp.add"}}}),
        "numpy.unverified": ("numpy.unverified", {"verified": False}),
        "numpy.nomap": ("numpy.nomap", {"verified": True, "variants": {}}),
      },
      {"jax": {"alias": {"module": "jax.numpy"}}, "torch": {"traits": {"module_base": "nn.Module"}}},
    )
    self.config = MockConfig()
    self.target_fw = "jax"
    self.source_fw = "numpy"
    self.strict_mode = True
    self.failures: list[str] = []

  def _report_failure(self, msg: str) -> None:
    """Fail."""
    self.failures.append(msg)


def test_cst_to_string() -> None:
  """Verifies the behavior of cst to string."""
  helper = MockHelper()
  assert helper._cst_to_string(cst.Name("foo")) == "foo"
  assert helper._cst_to_string(cst.Attribute(cst.Name("foo"), cst.Name("bar"))) == "foo.bar"
  node = cst.BinaryOperation(cst.Name("a"), cst.Add(), cst.Name("b"))
  assert helper._cst_to_string(node) == "Add"
  assert helper._cst_to_string(cst.Integer("1")) is None


def test_get_qualified_name() -> None:
  """Gets qualified name."""
  helper = MockHelper()
  assert helper._get_qualified_name(cst.Name("foo")) == "foo"
  assert helper._get_qualified_name(cst.Attribute(cst.Name("np"), cst.Name("add"))) == "numpy.add"
  assert helper._get_qualified_name(cst.Integer("1")) is None


def test_create_dotted_name() -> None:
  """Creates dotted name."""
  helper = MockHelper()
  node: typing.Any = helper._create_dotted_name("a.b.c")
  assert isinstance(node, cst.Attribute)
  assert node.attr.value == "c"


def test_is_module_alias() -> None:
  """Checks if is module alias."""
  helper = MockHelper()
  assert helper._is_module_alias(cst.Name("np")) is True
  assert helper._is_module_alias(cst.Name("torch")) is True
  assert helper._is_module_alias(cst.Name("jax")) is True
  assert helper._is_module_alias(cst.Name("unknown")) is False
  assert helper._is_module_alias(cst.Integer("1")) is False


def test_get_mapping() -> None:
  """Gets mapping."""
  helper = MockHelper()
  with pytest.MonkeyPatch().context() as m:
    import ml_switcheroo.core.rewriter.passes.api_helpers as helpers

    m.setattr(helpers, "get_tracer", lambda: MockTracer())
    mapping: typing.Any = helper._get_mapping("numpy.add")
    assert mapping is not None
    assert helper._get_mapping("numpy.missing") is None
    assert len(helper.failures) == 1
    assert helper._get_mapping("numpy.unverified") is None
    assert "Skipped" in helper.failures[1]
    assert helper._get_mapping("numpy.nomap") is None
    assert "No mapping" in helper.failures[2]


def test_handle_variant_imports() -> None:
  """Handles variant imports."""
  helper = MockHelper()
  variant: dict[str, typing.Any] = {
    "required_imports": ["import os", "sys", {"module": "math", "alias": "m"}, {"module": "json"}]
  }
  helper._handle_variant_imports(variant)
  stmts: list[typing.Any] = helper.context.hook_context.stmts
  assert stmts[0] == "import os"
  assert stmts[1] == "import sys"
  assert stmts[2] == "import math as m"
  assert stmts[3] == "import json"


def test_is_framework_base() -> None:
  """Checks if is framework base."""
  helper = MockHelper()
  assert helper._is_framework_base("nn.Module") is True
  assert helper._is_framework_base("Module") is True
  assert helper._is_framework_base("foo") is False
  assert helper._is_framework_base(None) is False  # type: ignore


def test_check_version_constraints() -> None:
  """Checks version constraints."""
  helper = MockHelper()
  assert helper.check_version_constraints(None, None) is None
  helper.semantics.framework_configs["jax"]["version"] = "1.5.0"
  assert helper.check_version_constraints("1.0", None) is None
  assert helper.check_version_constraints("2.0", None) is not None
  assert helper.check_version_constraints(None, "2.0") is None
  assert helper.check_version_constraints(None, "1.0") is not None
  helper.semantics.framework_configs["jax"].pop("version")
  with pytest.MonkeyPatch().context() as m:
    import importlib.metadata

    m.setattr(importlib.metadata, "version", lambda x: "2.0.0" if x == "jax" else None)
    assert helper.check_version_constraints("1.0", None) is None
    m.setattr(importlib.metadata, "version", lambda x: 1 / 0)
    assert helper.check_version_constraints("1.0", None) is None

    # test flax_nnx substitution
    helper.target_fw = "flax_nnx"
    m.setattr(importlib.metadata, "version", lambda x: "2.0.0" if x == "flax" else None)
    assert helper.check_version_constraints("1.0", None) is None


def test_apply_preamble() -> None:
  """Applies preamble."""
  helper = MockHelper()
  func = typing.cast(cst.FunctionDef, cst.parse_module("def foo():\n  '''doc'''\n  pass").body[0])
  new_func: typing.Any = helper._apply_preamble(func, ["x = 1"])
  assert isinstance(new_func.body.body[1], cst.SimpleStatementLine)
  assign = typing.cast(cst.Assign, new_func.body.body[1].body[0])
  assert typing.cast(cst.Name, assign.targets[0].target).value == "x"


def test_inject_argument() -> None:
  """Injects argument."""
  helper = MockHelper()
  func = typing.cast(cst.FunctionDef, cst.parse_module("def foo(self, a):\n  pass").body[0])
  new_func: typing.Any = helper._inject_argument_to_signature(func, "b", "int")
  assert len(new_func.params.params) == 3
  assert new_func.params.params[1].name.value == "b"
  assert new_func.params.params[1].annotation.annotation.value == "int"


def test_inject_argument_exists() -> None:
  """Injects argument exists."""
  helper = MockHelper()
  func = typing.cast(cst.FunctionDef, cst.parse_module("def foo(self, b):\n  pass").body[0])
  new_func: typing.Any = helper._inject_argument_to_signature(func, "b", "int")
  assert len(new_func.params.params) == 2


def test_apply_preamble_exception() -> None:
  """Tests fallback when cst.parse_module throws an error inside _apply_preamble."""
  helper = MockHelper()
  func = typing.cast(cst.FunctionDef, cst.parse_module("def foo():\n  pass").body[0])
  # Using invalid syntax to trigger parse error
  new_func: typing.Any = helper._apply_preamble(func, ["x = 1", "invalid syntax!"])
  assert len(new_func.body.body) == 2  # pass, and x=1


def test_convert_to_indented_block_fallback() -> None:
  """Tests _convert_to_indented_block when body is already IndentedBlock."""
  helper = MockHelper()
  func = typing.cast(cst.FunctionDef, cst.parse_module("def foo():\n  pass").body[0])
  # Function body is an IndentedBlock
  res: typing.Any = helper._convert_to_indented_block(func)  # type: ignore
  assert res is func


def test_get_mapping_not_dict() -> None:
  """Tests _get_mapping when target_impl is present but not a dict."""
  helper = MockHelper()

  class FakeImpl:
    """A fake implementation object."""

    def get(self, *args: typing.Any, **kwargs: typing.Any) -> str:
      """Gets fake api."""
      return "fake_api"

  helper.semantics.defs["numpy.stringmap"] = ("numpy.stringmap", {"verified": True, "variants": {"jax": FakeImpl()}})
  assert helper._get_mapping("numpy.stringmap") is None


def test_cst_to_string_fallback() -> None:
  """Tests _cst_to_string with unsupported node type."""
  helper = MockHelper()
  # Pass() is not supported, should return None
  assert helper._cst_to_string(cst.Pass()) is None


def test_version_parse_empty() -> None:
  """Tests check_version_constraints with an empty version string which raises error during parse."""
  helper = MockHelper()
  helper.semantics.framework_configs["jax"]["version"] = ""
  assert helper.check_version_constraints("1.0", None) is None
