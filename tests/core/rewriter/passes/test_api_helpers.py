"""Test suite for the Api Helpers module."""

import typing

import libcst as cst
import pytest
from pydantic import JsonValue

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.core.rewriter.passes.api_helpers import ApiPassState, MappingResolver
from ml_switcheroo.enums import RuntimeTensorOrValue
from ml_switcheroo.semantics.manager import SemanticsManager


class MockTracer:
  """Test double `MockTracer` implementing `log_match` for API helpers tests."""

  def log_match(self, *args: JsonValue, **kwargs: JsonValue) -> None:
    """Mock implementation of log match.

    Args:
        args: Positional argument specifications (`JsonValue`).
        kwargs: Keyword argument bindings (`JsonValue`).
    """


class MockSemantics(SemanticsManager):
  """Test double `MockSemantics` implementing `get_definition`, `is_verified` for API helpers tests."""

  def __init__(self, defs: dict[str, tuple[str, dict[str, JsonValue]]], configs: dict[str, dict[str, JsonValue]]) -> None:
    """Initialize `MockSemantics` with custom definition and framework dictionaries.

    Args:
        defs: Input `defs` parameter value (`dict[str, tuple[str, dict[str, JsonValue]]]`).
        configs: Input `configs` parameter value (`dict[str, dict[str, JsonValue]]`).
    """
    super().__init__(load_defaults=False)
    self.defs = defs
    self.framework_configs = configs
    self._key_origins = {}

  def get_definition(self, name: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Mock implementation of get definition.

    Args:
        name: Identifier or operation name string.

    Returns:
        Resulting `tuple[str, dict[str, JsonValue]] | None` from get definition.
    """
    return self.defs.get(name)

  def is_verified(self, name: str) -> bool:
    """Mock implementation of is verified.

    Args:
        name: Identifier or operation name string.

    Returns:
        Resulting `bool` from is verified.
    """
    raw = self.defs.get(name, ("", {"verified": True}))[1].get("verified", True)
    return bool(raw)

  def resolve_variant(self, abstract_id: str, target: str) -> dict[str, JsonValue] | None:
    """Mock implementation of resolve variant.

    Args:
        abstract_id: Canonical abstract operation identifier (`str`).
        target: Target framework or dialect identifier string.

    Returns:
        Resulting `dict[str, JsonValue] | None` from resolve variant.
    """
    variants = self.defs.get(abstract_id, ("", {}))[1].get("variants", {})
    if not isinstance(variants, dict):
      return None
    val = variants.get(target)
    return val if isinstance(val, dict) else None

  def get_framework_config(self, target: str) -> dict[str, JsonValue]:
    """Mock implementation of get framework configuration.

    Args:
        target: Target framework or dialect identifier string.

    Returns:
        Resulting `dict[str, JsonValue]` from get framework config.
    """
    return self.framework_configs.get(target, {})


class MockHookContext(HookContext):
  """Test double `MockHookContext` implementing `inject_preamble` for API helpers tests."""

  def __init__(self) -> None:
    """Initialize `MockHookContext` with an empty preamble list."""
    super().__init__(semantics=None, config=RuntimeConfig())
    self.stmts: list[RuntimeTensorOrValue] = []

  def inject_preamble(self, s: str) -> None:
    """Record injected preamble line into `self.stmts`.

    Args:
        s: Input `s` parameter value (`str`).
    """
    self.stmts.append(s)


class MockContext(RewriterContext):
  """Test double `MockContext` subclassing `RewriterContext` for API helpers tests."""

  hook_context: MockHookContext

  def __init__(self) -> None:
    """Initialize `MockContext` with numpy alias mapping and `MockHookContext`."""
    super().__init__(semantics=SemanticsManager.empty(), config=RuntimeConfig())
    self.alias_map = {"np": "numpy"}
    self.hook_context = MockHookContext()


def MockConfig() -> RuntimeConfig:
  """Construct a `RuntimeConfig` fixture for API helpers tests.

  Returns:
      Configured `RuntimeConfig` instance.
  """
  return RuntimeConfig(source_framework="torch", target_framework="jax", source_flavour="torch.nn")


class _MutableHelperState(ApiPassState):
  """Mutable `ApiPassState` subclass allowing property overrides in isolated `MappingResolver` tests."""

  context: MockContext

  def __init__(
    self,
    context: MockContext,
    semantics: MockSemantics,
    config: RuntimeConfig,
    source_framework: str,
    target_framework: str,
    strict_mode: bool,
  ) -> None:
    """Initialize mutable state attributes.

    Args:
        context: Mock context object.
        semantics: Mock semantics manager.
        config: Runtime config.
        source_framework: Source framework identifier.
        target_framework: Target framework identifier.
        strict_mode: Whether strict mode is enabled.
    """
    super().__init__(context=context)
    self._semantics_override: MockSemantics = semantics
    self._config_override: RuntimeConfig = config
    self._source_fw: str = source_framework
    self._target_fw: str = target_framework
    self._strict_flag: bool = strict_mode
    self.failures: list[str] = []

  @property
  def semantics(self) -> MockSemantics:
    """Return overridden semantics instance.

    Returns:
        Active `MockSemantics` instance.
    """
    return self._semantics_override

  @semantics.setter
  def semantics(self, val: MockSemantics) -> None:
    """Update overridden semantics instance.

    Args:
        val: Replacement `MockSemantics` instance.
    """
    self._semantics_override = val

  @property
  def config(self) -> RuntimeConfig:
    """Return overridden runtime configuration.

    Returns:
        Active `RuntimeConfig` instance.
    """
    return self._config_override

  @config.setter
  def config(self, val: RuntimeConfig) -> None:
    """Update overridden runtime configuration.

    Args:
        val: Replacement `RuntimeConfig` instance.
    """
    self._config_override = val

  @property
  def source_framework(self) -> str:
    """Return mutable source framework identifier.

    Returns:
        Source framework string.
    """
    return self._source_fw

  @source_framework.setter
  def source_framework(self, val: str) -> None:
    """Update mutable source framework identifier.

    Args:
        val: Replacement source framework string.
    """
    self._source_fw = val

  @property
  def target_framework(self) -> str:
    """Return mutable target framework identifier.

    Returns:
        Target framework string.
    """
    return self._target_fw

  @target_framework.setter
  def target_framework(self, val: str) -> None:
    """Update mutable target framework identifier.

    Args:
        val: Replacement target framework string.
    """
    self._target_fw = val

  @property
  def strict_mode(self) -> bool:
    """Return mutable strict mode flag.

    Returns:
        Strict mode boolean.
    """
    return self._strict_flag

  @strict_mode.setter
  def strict_mode(self, val: bool) -> None:
    """Update mutable strict mode flag.

    Args:
        val: Replacement strict mode flag.
    """
    self._strict_flag = val

  def report_failure(self, msg: str) -> None:
    """Record ``msg`` into ``self.failures``.

    Args:
        msg: Failure message string.
    """
    self.failures.append(msg)


class MockHelper:
  """Test harness delegating to ``MappingResolver``."""

  def __init__(self) -> None:
    """Initialize ``MockHelper`` with ``_MutableHelperState`` and ``MappingResolver``."""
    self._state = _MutableHelperState(
      context=MockContext(),
      semantics=MockSemantics(
        {
          "numpy.add": ("numpy.add", {"verified": True, "variants": {"jax": {"api": "jnp.add"}}}),
          "numpy.unverified": ("numpy.unverified", {"verified": False}),
          "numpy.nomap": ("numpy.nomap", {"verified": True, "variants": {}}),
        },
        {"jax": {"alias": {"module": "jax.numpy"}}, "torch": {"traits": {"module_base": "nn.Module"}}},
      ),
      config=MockConfig(),
      source_framework="numpy",
      target_framework="jax",
      strict_mode=True,
    )
    self.resolver = MappingResolver(self._state)

  @property
  def context(self) -> MockContext:
    """Return mock context.

    Returns:
        Bound ``MockContext``.
    """
    return self._state.context

  @property
  def semantics(self) -> MockSemantics:
    """Return mock semantics.

    Returns:
        Bound ``MockSemantics``.
    """
    return self._state.semantics

  @semantics.setter
  def semantics(self, val: MockSemantics) -> None:
    """Update mock semantics.

    Args:
        val: Replacement ``MockSemantics``.
    """
    self._state.semantics = val

  @property
  def config(self) -> RuntimeConfig:
    """Return mock config.

    Returns:
        Bound ``RuntimeConfig``.
    """
    return self._state.config

  @config.setter
  def config(self, val: RuntimeConfig) -> None:
    """Update mock config.

    Args:
        val: Replacement ``RuntimeConfig``.
    """
    self._state.config = val

  @property
  def target_framework(self) -> str:
    """Return target framework name.

    Returns:
        Target framework identifier.
    """
    return self._state.target_framework

  @target_framework.setter
  def target_framework(self, val: str) -> None:
    """Update target framework name.

    Args:
        val: Replacement target framework identifier.
    """
    self._state.target_framework = val

  @property
  def source_framework(self) -> str:
    """Return source framework name.

    Returns:
        Source framework identifier.
    """
    return self._state.source_framework

  @source_framework.setter
  def source_framework(self, val: str) -> None:
    """Update source framework name.

    Args:
        val: Replacement source framework identifier.
    """
    self._state.source_framework = val

  @property
  def strict_mode(self) -> bool:
    """Return strict mode flag.

    Returns:
        Strict mode boolean.
    """
    return self._state.strict_mode

  @strict_mode.setter
  def strict_mode(self, val: bool) -> None:
    """Update strict mode flag.

    Args:
        val: Replacement strict mode flag.
    """
    self._state.strict_mode = val

  @property
  def _known_module_bases(self) -> set[str] | None:
    """Return cached module base names.

    Returns:
        Set of module base strings or ``None``.
    """
    return self._state.known_module_bases

  @_known_module_bases.setter
  def _known_module_bases(self, val: set[str] | None) -> None:
    """Set cached module base names.

    Args:
        val: Set of module base strings or ``None``.
    """
    self._state.known_module_bases = val

  @property
  def failures(self) -> list[str]:
    """Return recorded failure strings.

    Returns:
        List of recorded failure messages.
    """
    return self._state.failures

  def _report_failure(self, msg: str) -> None:
    """Record failure message.

    Args:
        msg: Failure message string.
    """
    self._state.report_failure(msg)

  def _cst_to_string(self, node: cst.CSTNode) -> str | None:
    """Delegate CST expression flattening to ``self.resolver``.

    Args:
        node: CST node to flatten.

    Returns:
        Dotted string or ``None``.
    """
    return self.resolver.cst_to_string(node)

  def _get_qualified_name(self, node: cst.BaseExpression) -> str | None:
    """Delegate FQN lookup to ``self.resolver``.

    Args:
        node: CST expression node.

    Returns:
        Qualified name string or ``None``.
    """
    return self.resolver.get_qualified_name(node)

  def _create_name_node(self, api_path: str) -> cst.Name | cst.Attribute:
    """Delegate CST ``Name``/``Attribute`` construction to ``self.resolver``.

    Args:
        api_path: Dotted API path.

    Returns:
        Constructed ``cst.Name`` or ``cst.Attribute``.
    """
    return self.resolver.create_name_node(api_path)

  def _create_dotted_name(self, name_str: str) -> cst.Name | cst.Attribute:
    """Delegate dotted CST expression construction to ``self.resolver``.

    Args:
        name_str: Dotted name string.

    Returns:
        Constructed ``cst.Name`` or ``cst.Attribute``.
    """
    return self.resolver.create_dotted_name(name_str)

  def _is_module_alias(self, node: cst.CSTNode) -> bool:
    """Delegate module alias detection to ``self.resolver``.

    Args:
        node: CST node to inspect.

    Returns:
        ``True`` if ``node`` is a module alias.
    """
    return self.resolver.is_module_alias(node)

  def _apply_preamble(self, node: cst.FunctionDef, stmts_code: list[str]) -> cst.FunctionDef:
    """Delegate preamble injection to ``self.resolver``.

    Args:
        node: FunctionDef node.
        stmts_code: Statement strings to inject.

    Returns:
        Updated ``cst.FunctionDef``.
    """
    return self.resolver.apply_preamble(node, stmts_code)

  def _inject_stmts_to_body(
    self,
    node: cst.FunctionDef,
    new_stmts: list[cst.SimpleStatementLine | cst.BaseCompoundStatement],
  ) -> cst.FunctionDef:
    """Delegate statement insertion to ``self.resolver``.

    Args:
        node: FunctionDef node.
        new_stmts: CST statements to insert.

    Returns:
        Updated ``cst.FunctionDef``.
    """
    return self.resolver.inject_stmts_to_body(node, new_stmts)

  def _convert_to_indented_block(self, node: cst.FunctionDef) -> cst.FunctionDef:
    """Delegate suite normalization to ``self.resolver``.

    Args:
        node: FunctionDef node.

    Returns:
        Normalized ``cst.FunctionDef``.
    """
    return self.resolver.convert_to_indented_block(node)

  def _get_mapping(self, name: str, silent: bool = False) -> dict[str, JsonValue] | None:
    """Delegate variant mapping lookup to ``self.resolver``.

    Args:
        name: Source API name.
        silent: Whether to silence failure reporting.

    Returns:
        Target variant mapping dict or ``None``.
    """
    return self.resolver.get_mapping(name, silent=silent, report_failure_fn=self._report_failure)

  def _handle_variant_imports(self, variant: typing.Mapping[str, RuntimeTensorOrValue]) -> None:
    """Delegate variant import injection to ``self.resolver``.

    Args:
        variant: Variant dictionary.
    """
    self.resolver.handle_variant_imports(variant)

  def _is_framework_base(self, name: str) -> bool:
    """Delegate framework module base check to ``self.resolver``.

    Args:
        name: Candidate class name.

    Returns:
        ``True`` if ``name`` is a framework module base.
    """
    return self.resolver.is_framework_base(name)

  def check_version_constraints(self, min_v: str | None, max_v: str | None) -> str | None:
    """Delegate version constraint check to ``self.resolver``.

    Args:
        min_v: Minimum version string or ``None``.
        max_v: Maximum version string or ``None``.

    Returns:
        Warning string or ``None``.
    """
    return self.resolver.check_version_constraints(min_v, max_v)

  def _inject_argument_to_signature(
    self,
    node: cst.FunctionDef,
    arg_name: str,
    annotation: str | None,
  ) -> cst.FunctionDef:
    """Delegate parameter injection to ``self.resolver``.

    Args:
        node: FunctionDef node.
        arg_name: Parameter name.
        annotation: Optional type annotation string.

    Returns:
        Updated ``cst.FunctionDef``.
    """
    return self.resolver.inject_argument_to_signature(node, arg_name, annotation)


def test_cst_to_string() -> None:
  """Verify `BinaryOperation` handles CST to string as expected."""
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


def test_get_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
  """Gets mapping.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
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
    monkeypatch.setattr(
      helper,
      "target_framework",
      "NumPy",
      raising=False,
    )
    helper.semantics._key_origins = {"numpy.nomap": "neural"}
    assert helper._get_mapping("numpy.nomap") is None
    assert "Cannot map neural network abstraction" in helper.failures[-1]


def test_handle_variant_imports() -> None:
  """Handles variant imports."""
  helper = MockHelper()
  variant = {"required_imports": ["import os", "sys", {"module": "math", "alias": "m"}, {"module": "json"}]}
  helper._handle_variant_imports(variant)
  stmts = helper.context.hook_context.stmts
  assert stmts[0] == "import os"
  assert stmts[1] == "import sys"
  assert stmts[2] == "import math as m"
  assert stmts[3] == "import json"


def test_is_framework_base(monkeypatch: pytest.MonkeyPatch) -> None:
  """Checks if is framework base.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  helper = MockHelper()
  # Add a framework config with traits that have no module_base
  helper.semantics.framework_configs["fw_no_base"] = {"traits": {"other": "val"}}
  monkeypatch.setattr(helper, "_known_module_bases", None, raising=False)
  assert helper._is_framework_base("nn.Module") is True
  assert helper._is_framework_base("Module") is True
  assert helper._is_framework_base("foo") is False
  assert helper._is_framework_base("") is False


def test_check_version_constraints(monkeypatch: pytest.MonkeyPatch) -> None:
  """Checks version constraints.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
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

    def _missing_pkg(x: str) -> str:
      """Simulate missing package metadata.

      Args:
          x: Package name.

      Returns:
          Never returns normally.

      Raises:
          importlib.metadata.PackageNotFoundError: Always raised.
      """
      raise importlib.metadata.PackageNotFoundError(x)

    m.setattr(importlib.metadata, "version", _missing_pkg)
    assert helper.check_version_constraints("1.0", None) is None

    # test flax_nnx substitution
    monkeypatch.setattr(
      helper,
      "target_framework",
      "flax_nnx",
      raising=False,
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
  param_ann = new_func.params.params[1].annotation
  assert param_ann is not None
  assert cst.ensure_type(param_ann.annotation, cst.Name).value == "int"


def test_inject_argument_exists() -> None:
  """Injects argument exists."""
  helper = MockHelper()
  func = cst.ensure_type(cst.parse_module("def foo(self, b):\n  pass").body[0], cst.FunctionDef)
  new_func = helper._inject_argument_to_signature(func, "b", "int")
  assert len(new_func.params.params) == 2


def test_apply_preamble_exception() -> None:
  """Verify `ensure_type` returns `2` for apply preamble exception."""
  helper = MockHelper()
  func = cst.ensure_type(cst.parse_module("def foo():\n  pass").body[0], cst.FunctionDef)
  # Using invalid syntax to trigger parse error
  new_func = helper._apply_preamble(func, ["x = 1", "invalid syntax!"])
  assert len(new_func.body.body) == 2  # pass, and x=1


def test_convert_to_indented_block_fallback() -> None:
  """Verify `ensure_type` handles convert to indented block fallback as expected."""
  helper = MockHelper()
  func = cst.ensure_type(cst.parse_module("def foo():\n  pass").body[0], cst.FunctionDef)
  # Function body is an IndentedBlock
  res = helper._convert_to_indented_block(func)
  assert res is func


def test_get_mapping_not_dict() -> None:
  """Verify `helper._get_mapping` handles non-dictionary variant payloads safely."""
  helper = MockHelper()
  helper.semantics.defs["numpy.stringmap"] = (
    "numpy.stringmap",
    {
      "verified": True,
      "variants": {
        "jax": "not_a_dict",
      },
    },
  )
  assert helper._get_mapping("numpy.stringmap") is None


def test_cst_to_string_fallback() -> None:
  """Verify `helper._cst_to_string` handles fallback as expected."""
  helper = MockHelper()
  # Pass() is not supported, should return None
  assert helper._cst_to_string(cst.Pass()) is None


def test_version_parse_empty() -> None:
  """Verify `helper.check_version_constraints` handles version parse empty as expected."""
  helper = MockHelper()
  helper.semantics.framework_configs["jax"]["version"] = ""
  assert helper.check_version_constraints("1.0", None) is None


def test_api_helpers_branch_coverage_extensions(monkeypatch: pytest.MonkeyPatch) -> None:
  """Test branch gaps in ApiHelpersMixin.

  Args:
      monkeypatch: Pytest monkeypatch fixture for modifying attributes or environment.
  """
  helper = MockHelper()

  # 1. 49->54: _cst_to_string on attribute with non-flattenable base
  attr_call = cst.Attribute(value=cst.Call(func=cst.Name("foo")), attr=cst.Name("bar"))
  assert helper._cst_to_string(attr_call) is None

  # 2. 124->130, 130->140, 137->132: _is_module_alias branches
  node_mod = cst.Name("other_mod")
  # 124->130: self.config is None
  monkeypatch.setattr(helper._state, "_config_override", None)
  assert not helper._is_module_alias(node_mod)

  # 137->132: alias_conf is dict but module is None
  monkeypatch.setattr(helper._state, "_config_override", MockConfig())
  helper.semantics.framework_configs = {"torch": {"alias": {"module": None}}}
  assert not helper._is_module_alias(node_mod)

  # 130->140: self.semantics is None
  monkeypatch.setattr(helper._state, "_semantics_override", None)
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
  assert "import os" in helper.context.hook_context.stmts

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
          Resulting `bool` from bool.
      """
      return False

  monkeypatch.setattr("ml_switcheroo.core.rewriter.passes.api_helpers.list", _AlwaysFalseList, raising=False)
  func_base = cst.ensure_type(cst.parse_module("def foo():\n  pass").body[0], cst.FunctionDef)
  res_falsy = helper._inject_argument_to_signature(func_base, "z", None)
  assert len(res_falsy.params.params) >= 1


def test_merged_api_helpers_short_fqn_simple_suite_and_non_list_imports() -> None:
  """Verify short root alias FQN resolution, one-liner suite injection, self comma, and non-list imports."""
  from ml_switcheroo.config import RuntimeConfig
  from ml_switcheroo.core.rewriter.context import RewriterContext
  from ml_switcheroo.semantics.manager import SemanticsManager

  helper = MockHelper()
  helper.context.alias_map["my_root"] = "canonical_root"
  assert helper._get_qualified_name(cst.Name("my_root")) == "canonical_root"

  one_liner = cst.ensure_type(cst.parse_statement("def foo(): pass"), cst.FunctionDef)
  injected = helper._inject_stmts_to_body(one_liner, [cst.SimpleStatementLine(body=[cst.Pass()])])
  assert isinstance(injected.body, cst.IndentedBlock)
  assert len(injected.body.body) == 2

  with_self = cst.ensure_type(cst.parse_statement("def foo(self): pass"), cst.FunctionDef)
  sig_updated = helper._inject_argument_to_signature(with_self, "new_arg", None)
  params = list(sig_updated.params.params)
  assert len(params) == 2
  assert params[0].name.value == "self"
  assert isinstance(params[0].comma, cst.Comma)
  assert params[1].name.value == "new_arg"

  helper._handle_variant_imports({"required_imports": "not_a_list"})

  state = ApiPassState(context=RewriterContext(semantics=SemanticsManager.empty(), config=RuntimeConfig()))
  assert state.source_traits is state.source_traits
  assert state.target_traits is state.target_traits
