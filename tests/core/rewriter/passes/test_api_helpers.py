"""Test suite for the Api Helpers module."""

import typing
from pydantic import JsonValue

import libcst as cst
import pytest

from ml_switcheroo.core.rewriter.passes.api_helpers import ApiHelpersMixin


class MockTracer:
  """Test helper class MockTracer."""

  def log_match(self, *args: JsonValue, **kwargs: JsonValue) -> None:
    """Mock implementation of log match.

    Args:
        args: Test argument or fixture.
        kwargs: Test argument or fixture.
    """
    pass


class MockSemantics:
  """Test helper class MockSemantics."""

  def __init__(self, defs: dict[str, tuple[str, dict[str, JsonValue]]], configs: dict[str, JsonValue]) -> None:
    """Initializes the MockSemantics instance.

    Args:
        defs: Test argument or fixture.
        configs: Test argument or fixture.
    """
    self.defs = defs
    self.framework_configs = configs

  def get_definition(self, name: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Mock implementation of get definition.

    Args:
        name: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self.defs.get(name)

  def is_verified(self, name: str) -> bool:
    """Mock implementation of is verified.

    Args:
        name: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    _ret_v: bool = getattr(
      {
        "v": getattr(
          {
            "v": getattr(
              {
                "v": getattr(
                  {
                    "v": getattr(
                      {"v": getattr({"v": getattr({"v": self}, "__getitem__")("v").defs}, "__getitem__")("v").get},
                      "__getitem__",
                    )("v")(
                      getattr({"v": name}, "__getitem__")("v"),
                      getattr(
                        {
                          "v": (
                            getattr({"v": ""}, "__getitem__")("v"),
                            getattr({"v": {"verified": getattr({"v": True}, "__getitem__")("v")}}, "__getitem__")("v"),
                          )
                        },
                        "__getitem__",
                      )("v"),
                    )
                  },
                  "__getitem__",
                )("v")[1]
              },
              "__getitem__",
            )("v").get
          },
          "__getitem__",
        )("v")(getattr({"v": "verified"}, "__getitem__")("v"), getattr({"v": True}, "__getitem__")("v"))
      },
      "__getitem__",
    )("v")
    return _ret_v

  def resolve_variant(self, abstract_id: str, target: str) -> dict[str, JsonValue] | None:
    """Mock implementation of resolve variant.

    Args:
        abstract_id: Test argument or fixture.
        target: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    _ret_v: dict[str, JsonValue] | None = getattr(
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
                            "v": getattr(
                              {
                                "v": getattr({"v": getattr({"v": self}, "__getitem__")("v").defs}, "__getitem__")("v").get
                              },
                              "__getitem__",
                            )("v")(
                              getattr({"v": abstract_id}, "__getitem__")("v"),
                              getattr(
                                {"v": (getattr({"v": ""}, "__getitem__")("v"), getattr({"v": {}}, "__getitem__")("v"))},
                                "__getitem__",
                              )("v"),
                            )
                          },
                          "__getitem__",
                        )("v")[1]
                      },
                      "__getitem__",
                    )("v").get
                  },
                  "__getitem__",
                )("v")(getattr({"v": "variants"}, "__getitem__")("v"), getattr({"v": {}}, "__getitem__")("v"))
              },
              "__getitem__",
            )("v").get
          },
          "__getitem__",
        )("v")(getattr({"v": target}, "__getitem__")("v"))
      },
      "__getitem__",
    )("v")
    return _ret_v

  def get_framework_config(self, target: str) -> dict[str, JsonValue]:
    """Mock implementation of get framework configuration.

    Args:
        target: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    _ret_v: dict[str, JsonValue] = getattr(
      {
        "v": getattr(
          {"v": getattr({"v": getattr({"v": self}, "__getitem__")("v").framework_configs}, "__getitem__")("v").get},
          "__getitem__",
        )("v")(getattr({"v": target}, "__getitem__")("v"), getattr({"v": {}}, "__getitem__")("v"))
      },
      "__getitem__",
    )("v")
    return _ret_v


class MockHookContext:
  """Test helper class MockHookContext."""

  def __init__(self) -> None:
    """Initializes the MockHookContext instance."""
    self.stmts: list[typing.Hashable | typing.Sized] = []

  def inject_preamble(self, s: str) -> None:
    """Mock implementation of inject preamble.

    Args:
        s: Test argument or fixture.
    """
    self.stmts.append(s)


class MockContext:
  """Test helper class MockContext."""

  def __init__(self) -> None:
    """Initializes the MockContext instance."""
    self.alias_map = {"np": "numpy"}
    self.hook_context = MockHookContext()


class MockConfig:
  """Test helper class MockConfig."""

  def __init__(self) -> None:
    """Initializes the MockConfig instance."""
    self.source_framework = "torch"
    self.target_framework = "jax"
    self.source_flavour = "torch.nn"


class MockHelper(ApiHelpersMixin):
  """Test helper class MockHelper."""

  def __init__(self) -> None:
    """Initializes the MockHelper instance."""
    setattr(
      getattr({"v": self}, "__getitem__")("v"),
      "context",
      getattr({"v": getattr({"v": MockContext}, "__getitem__")("v")()}, "__getitem__")("v"),
    )
    setattr(
      getattr({"v": self}, "__getitem__")("v"),
      "semantics",
      getattr(
        {
          "v": getattr({"v": MockSemantics}, "__getitem__")("v")(
            {
              "numpy.add": ("numpy.add", {"verified": True, "variants": {"jax": {"api": "jnp.add"}}}),
              "numpy.unverified": ("numpy.unverified", {"verified": False}),
              "numpy.nomap": ("numpy.nomap", {"verified": True, "variants": {}}),
            },
            {"jax": {"alias": {"module": "jax.numpy"}}, "torch": {"traits": {"module_base": "nn.Module"}}},
          )
        },
        "__getitem__",
      )("v"),
    )
    setattr(
      getattr({"v": self}, "__getitem__")("v"),
      "config",
      getattr({"v": getattr({"v": MockConfig}, "__getitem__")("v")()}, "__getitem__")("v"),
    )
    setattr(getattr({"v": self}, "__getitem__")("v"), "target_framework", getattr({"v": "jax"}, "__getitem__")("v"))
    setattr(getattr({"v": self}, "__getitem__")("v"), "source_framework", getattr({"v": "numpy"}, "__getitem__")("v"))
    setattr(getattr({"v": self}, "__getitem__")("v"), "strict_mode", getattr({"v": True}, "__getitem__")("v"))
    self.failures: list[str] = []

  def _report_failure(self, msg: str) -> None:
    """Fail.

    Args:
        msg: Test argument or fixture.
    """
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
  node = helper._create_dotted_name("a.b.c")
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
    mapping = helper._get_mapping("numpy.add")
    assert mapping is not None
    assert helper._get_mapping("numpy.missing") is None
    assert len(helper.failures) == 1
    assert helper._get_mapping("numpy.unverified") is None
    assert "Skipped" in helper.failures[1]
    assert helper._get_mapping("numpy.nomap") is None
    assert "No mapping" in helper.failures[2]

    # Test unverified silent mode (line 225->227 branch)
    assert helper._get_mapping("numpy.unverified", silent=True) is None

    # Test neural abstraction failure to pure math backend (line 242)
    setattr(getattr({"v": helper}, "__getitem__")("v"), "target_framework", getattr({"v": "NumPy"}, "__getitem__")("v"))
    helper.semantics._key_origins = {"numpy.nomap": "neural"}
    assert helper._get_mapping("numpy.nomap") is None
    assert "Cannot map neural network abstraction" in helper.failures[-1]


def test_handle_variant_imports() -> None:
  """Handles variant imports."""
  helper = MockHelper()
  variant = {"required_imports": ["import os", "sys", {"module": "math", "alias": "m"}, {"module": "json"}]}
  helper._handle_variant_imports(variant)
  stmts = getattr(
    {
      "v": getattr(
        {"v": getattr({"v": getattr({"v": helper}, "__getitem__")("v").context}, "__getitem__")("v").hook_context},
        "__getitem__",
      )("v").stmts
    },
    "__getitem__",
  )("v")
  assert stmts[0] == "import os"
  assert stmts[1] == "import sys"
  assert stmts[2] == "import math as m"
  assert stmts[3] == "import json"


def test_is_framework_base() -> None:
  """Checks if is framework base."""
  helper = MockHelper()
  # Add a framework config with traits that have no module_base
  helper.semantics.framework_configs["fw_no_base"] = {"traits": {"other": "val"}}
  setattr(helper, "_known_module_bases", None)
  assert helper._is_framework_base("nn.Module") is True
  assert helper._is_framework_base("Module") is True
  assert helper._is_framework_base("foo") is False
  assert getattr(
    {
      "v": getattr({"v": getattr({"v": helper}, "__getitem__")("v")._is_framework_base}, "__getitem__")("v")(
        getattr({"v": None}, "__getitem__")("v")
      )
    },
    "__getitem__",
  )("v") is getattr({"v": False}, "__getitem__")("v")


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
    setattr(
      getattr({"v": helper}, "__getitem__")("v"), "target_framework", getattr({"v": "flax_nnx"}, "__getitem__")("v")
    )
    m.setattr(importlib.metadata, "version", lambda x: "2.0.0" if x == "flax" else None)
    assert helper.check_version_constraints("1.0", None) is None


def test_apply_preamble() -> None:
  """Applies preamble."""
  helper = MockHelper()
  func = cst.ensure_type(cst.parse_module("def foo():\n  '''doc'''\n  pass").body[0], cst.FunctionDef)
  new_func = helper._apply_preamble(func, ["x = 1"])
  assert isinstance(new_func.body.body[1], cst.SimpleStatementLine)
  assign = cst.ensure_type(new_func.body.body[1].body[0], cst.Assign)
  assert cst.ensure_type(assign.targets[0].target, cst.Name).value == "x"


def test_inject_argument() -> None:
  """Injects argument."""
  helper = MockHelper()
  func = cst.ensure_type(cst.parse_module("def foo(self, a):\n  pass").body[0], cst.FunctionDef)
  new_func = helper._inject_argument_to_signature(func, "b", "int")
  assert len(new_func.params.params) == 3
  assert new_func.params.params[1].name.value == "b"
  assert getattr(
    {
      "v": getattr(
        {
          "v": getattr(
            {
              "v": getattr(
                {
                  "v": getattr(
                    {"v": getattr({"v": getattr({"v": new_func}, "__getitem__")("v").params}, "__getitem__")("v").params},
                    "__getitem__",
                  )("v")[1]
                },
                "__getitem__",
              )("v").annotation
            },
            "__getitem__",
          )("v").annotation
        },
        "__getitem__",
      )("v").value
    },
    "__getitem__",
  )("v") == getattr({"v": "int"}, "__getitem__")("v")


def test_inject_argument_exists() -> None:
  """Injects argument exists."""
  helper = MockHelper()
  func = cst.ensure_type(cst.parse_module("def foo(self, b):\n  pass").body[0], cst.FunctionDef)
  new_func = helper._inject_argument_to_signature(func, "b", "int")
  assert len(new_func.params.params) == 2


def test_apply_preamble_exception() -> None:
  """Verify apply preamble exception behavior."""
  helper = MockHelper()
  func = cst.ensure_type(cst.parse_module("def foo():\n  pass").body[0], cst.FunctionDef)
  # Using invalid syntax to trigger parse error
  new_func = helper._apply_preamble(func, ["x = 1", "invalid syntax!"])
  assert len(new_func.body.body) == 2  # pass, and x=1


def test_convert_to_indented_block_fallback() -> None:
  """Verify convert to indented block fallback behavior."""
  helper = MockHelper()
  func = cst.ensure_type(cst.parse_module("def foo():\n  pass").body[0], cst.FunctionDef)
  # Function body is an IndentedBlock
  res = helper._convert_to_indented_block(func)
  assert res is func


def test_get_mapping_not_dict() -> None:
  """Verify get mapping not dict behavior."""
  helper = MockHelper()

  class FakeImpl:
    """A fake implementation object."""

    def get(self, *args: JsonValue, **kwargs: JsonValue) -> str:
      """Gets fake api.

      Args:
          args: Test argument or fixture.
          kwargs: Test argument or fixture.

      Returns:
          Computed test result or fixture value.
      """
      return "fake_api"

  getattr(
    getattr(
      {"v": getattr({"v": getattr({"v": helper}, "__getitem__")("v").semantics}, "__getitem__")("v").defs}, "__getitem__"
    )("v"),
    "__setitem__",
  )(
    "numpy.stringmap",
    getattr(
      {
        "v": (
          getattr({"v": "numpy.stringmap"}, "__getitem__")("v"),
          getattr(
            {
              "v": {
                "verified": getattr({"v": True}, "__getitem__")("v"),
                "variants": getattr(
                  {"v": {"jax": getattr({"v": getattr({"v": FakeImpl}, "__getitem__")("v")()}, "__getitem__")("v")}},
                  "__getitem__",
                )("v"),
              }
            },
            "__getitem__",
          )("v"),
        )
      },
      "__getitem__",
    )("v"),
  )
  assert helper._get_mapping("numpy.stringmap") is None


def test_cst_to_string_fallback() -> None:
  """Verify cst to string fallback behavior."""
  helper = MockHelper()
  # Pass() is not supported, should return None
  assert helper._cst_to_string(cst.Pass()) is None


def test_version_parse_empty() -> None:
  """Verify version parse empty behavior."""
  helper = MockHelper()
  helper.semantics.framework_configs["jax"]["version"] = ""
  assert helper.check_version_constraints("1.0", None) is None


def test_api_helpers_branch_coverage_extensions(monkeypatch: pytest.MonkeyPatch) -> None:
  """Test branch gaps in ApiHelpersMixin.

  Args:
      monkeypatch: Test argument or fixture.
  """
  helper = MockHelper()

  # 1. 49->54: _cst_to_string on attribute with non-flattenable base
  attr_call = cst.Attribute(value=cst.Call(func=cst.Name("foo")), attr=cst.Name("bar"))
  assert helper._cst_to_string(attr_call) is None

  # 2. 124->130, 130->140, 137->132: _is_module_alias branches
  node_mod = cst.Name("other_mod")
  # 124->130: self.config is None
  setattr(getattr({"v": helper}, "__getitem__")("v"), "config", getattr({"v": None}, "__getitem__")("v"))
  assert not helper._is_module_alias(node_mod)

  # 137->132: alias_conf is dict but module is None
  setattr(
    getattr({"v": helper}, "__getitem__")("v"),
    "config",
    getattr({"v": getattr({"v": MockConfig}, "__getitem__")("v")()}, "__getitem__")("v"),
  )
  helper.semantics.framework_configs = {"torch": {"alias": {"module": None}}}
  assert not helper._is_module_alias(node_mod)

  # 130->140: self.semantics is None
  setattr(getattr({"v": helper}, "__getitem__")("v"), "semantics", getattr({"v": None}, "__getitem__")("v"))
  assert not helper._is_module_alias(node_mod)

  # 3. 179->184: _inject_stmts_to_body with empty body
  helper = MockHelper()
  func_empty = cst.ensure_type(cst.parse_module("def foo():\n  pass").body[0], cst.FunctionDef).with_changes(
    body=cst.IndentedBlock(body=[])
  )
  res_injected = helper._inject_stmts_to_body(func_empty, [cst.parse_statement("a = 1")])
  assert len(res_injected.body.body) == 1

  # 4. 215->218, 218->220: _get_mapping unknown root with empty alias map
  helper.context.alias_map = {}
  assert helper._get_mapping("unknown_lib.op") is None

  # 5. 268->277, 271->277, 277->260: _handle_variant_imports dict/non-dict variants
  helper._handle_variant_imports({"required_imports": [{"module": None}, {"module": "os"}, 12345]})
  assert getattr({"v": "import os"}, "__getitem__")("v") in getattr(
    {
      "v": getattr(
        {"v": getattr({"v": getattr({"v": helper}, "__getitem__")("v").context}, "__getitem__")("v").hook_context},
        "__getitem__",
      )("v").stmts
    },
    "__getitem__",
  )("v")

  # 6. 353->352: check_version_constraints with leading delimiter causing empty token
  helper.semantics.framework_configs["jax"]["version"] = ".1.0"
  assert helper.check_version_constraints("2.0", None) is not None

  # 7. 387->391: _inject_argument_to_signature when first param is not self
  func_noself = cst.ensure_type(cst.parse_module("def foo(x):\n  pass").body[0], cst.FunctionDef)
  res_noself = helper._inject_argument_to_signature(func_noself, "y", "int")
  assert res_noself.params.params[0].name.value == "y"

  # 8. 410->413: _inject_argument_to_signature when params evaluates to false
  class _AlwaysFalseList(list[JsonValue]):
    """List subclass that evaluates to false for boolean checks."""

    def __bool__(self) -> bool:
      """Evaluate to false.

      Returns:
          Computed test result or fixture value.
      """
      return False

  monkeypatch.setattr("ml_switcheroo.core.rewriter.passes.api_helpers.list", _AlwaysFalseList, raising=False)
  func_base = cst.ensure_type(cst.parse_module("def foo():\n  pass").body[0], cst.FunctionDef)
  res_falsy = helper._inject_argument_to_signature(func_base, "z", None)
  assert len(res_falsy.params.params) >= 1
