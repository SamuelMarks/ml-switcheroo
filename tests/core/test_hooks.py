"""Unit tests for `HookContext`, `HookRegistry`, `AutoWireSpec`, and plugin loading (`ml_switcheroo.core.hooks`)."""

import sys
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import libcst as cst
import pytest
from pydantic import BaseModel, JsonValue, ValidationError

from ml_switcheroo.analysis.symbol_table import SymbolTable
from ml_switcheroo.analysis.symbol_types import SymbolType
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.dsl import FrameworkVariant, OperationDef
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.core.hooks import (
  AutoWireSpec,
  HookContext,
  HookRegistry,
  clear_hooks,
  get_default_hook_registry,
  get_hook,
  load_plugins,
  register_hook,
  set_plugins_loaded,
)
from ml_switcheroo.core.hooks_registry import _import_from_dir, bind_hook_registry, get_all_hook_metadata
from ml_switcheroo.exceptions import PluginLoadError
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.semantics.schema import PluginTraits


def _build_semantics_fixture() -> SemanticsManager:
  """Construct an in-memory `SemanticsManager` with `add`, `abs`, and `complex` ops.

  Returns:
      Populated `SemanticsManager` instance.
  """
  sem = SemanticsManager(load_defaults=False)
  defs: dict[str, dict[str, JsonValue]] = {
    "add": {
      "operation": "add",
      "std_args": ["x1", "x2"],
      "variants": {"jax": {"api": "jax.numpy.add", "pack_to_tuple": "axes"}, "numpy": {"api": "numpy.add"}},
    },
    "abs": {
      "operation": "abs",
      "std_args": [["x", "Array"], {"name": "out", "type": "Array"}, {"type": "NoName"}, "plain_arg"],
      "variants": {},
    },
    "bad_sig": {
      "operation": "bad_sig",
      "std_args": "not_a_list",
      "variants": {},
    },
    "complex": {
      "operation": "complex",
      "std_args": [],
      "variants": {"jax": {"requires_plugin": "magic"}},
    },
  }
  for op_id, raw in defs.items():
    sem.data[op_id] = raw
    if op_id not in ("abs", "bad_sig"):
      sem.operations[op_id] = OperationDef.model_validate(raw)
  return sem


def test_autowire_spec_and_metadata_isolation() -> None:
  """Verify `AutoWireSpec` storage and `HookContext.metadata` per-instance isolation."""
  spec = AutoWireSpec(ops={"test_op": {"api": "test_api"}})
  assert spec.ops is not None
  op_entry = spec.ops["test_op"]
  assert isinstance(op_entry, dict)
  assert op_entry["api"] == "test_api"

  sem = _build_semantics_fixture()
  cfg = RuntimeConfig(source_framework="torch", target_framework="jax", plugin_settings={"any_key": 123})
  ctx1 = HookContext(sem, cfg)
  ctx2 = HookContext(sem, cfg)
  ctx1.metadata["scope_id"] = 1
  nested: dict[str, JsonValue] = {"flag": True}
  ctx1.metadata["plugin_data"] = nested

  assert "scope_id" not in ctx2.metadata
  assert "plugin_data" not in ctx2.metadata
  assert ctx1.raw_config("any_key", default=999) == 123
  assert ctx1.raw_config("missing_key", default="fallback") == "fallback"


def test_hook_context_symbol_table_traits_variants_and_settings() -> None:
  """Verify `resolve_type`, `plugin_traits`, `current_variant`, `lookup_api`, `lookup_signature`, and settings."""
  sem = _build_semantics_fixture()
  sym_table = SymbolTable()
  t_node = cst.Name("t")
  m_node = cst.Name("m")
  i_node = cst.Name("i")
  u_node = cst.Name("u")
  sym_table.record_type(t_node, SymbolType("torch.Tensor"))
  sym_table.record_type(m_node, SymbolType("torch.nn.Module"))
  sym_table.record_type(i_node, SymbolType("int"))

  injected_args: list[tuple[str, str | None]] = []
  injected_preambles: list[str] = []
  ctx = HookContext(
    semantics=sem,
    config=RuntimeConfig(
      source_framework="torch",
      target_framework="jax",
      plugin_settings={"epsilon": 0.001, "extra_ignored": True},
    ),
    arg_injector=lambda name, ann: injected_args.append((name, ann)),
    preamble_injector=injected_preambles.append,
    symbol_table=sym_table,
  )

  # Symbol table resolution and missing symbol table fallback
  assert ctx.resolve_type(t_node) == "Tensor"
  assert ctx.resolve_type(m_node) == "Module"
  assert ctx.resolve_type(i_node) == "int"
  assert ctx.resolve_type(u_node) is None

  no_sym_ctx = HookContext(semantics=None, config=RuntimeConfig(target_framework="jax"))
  assert no_sym_ctx.resolve_type(t_node) is None

  # Injectors present vs None
  ctx.inject_signature_arg("rng", "jax.Array")
  ctx.inject_preamble("import math")
  assert injected_args == [("rng", "jax.Array")]
  assert injected_preambles == ["import math"]
  no_sym_ctx.inject_signature_arg("x", "int")
  no_sym_ctx.inject_preamble("import os")

  # Settings validation success and failure
  class _PluginSchema(BaseModel):
    """Pydantic plugin settings schema."""

    epsilon: float = 1e-5

  validated = ctx.validate_settings(_PluginSchema)
  assert validated.epsilon == 0.001
  assert "extra_ignored" not in type(validated).model_fields

  bad_ctx = HookContext(sem, RuntimeConfig(plugin_settings={"epsilon": "not_a_float"}))
  with pytest.raises(ValidationError):
    bad_ctx.validate_settings(_PluginSchema)

  # `plugin_traits` across dict, PluginTraits instance, unsupported value, empty dict, None config, None semantics
  assert no_sym_ctx.plugin_traits.requires_explicit_rng is False
  sem.framework_configs["jax"] = {"plugin_traits": {"requires_explicit_rng": True, "has_numpy_compatible_arrays": True}}
  assert ctx.plugin_traits.requires_explicit_rng is True
  assert ctx.plugin_traits.has_numpy_compatible_arrays is True

  explicit_traits = PluginTraits(requires_explicit_rng=True)
  mock_sem = MagicMock(spec=SemanticsManager)
  mock_sem.get_framework_config.return_value = {"plugin_traits": explicit_traits}
  assert HookContext(mock_sem, RuntimeConfig(target_framework="jax")).plugin_traits is explicit_traits

  raw_cfgs: list[dict[str, JsonValue] | None] = [{"plugin_traits": "unsupported"}, {"plugin_traits": {}}, {}, None]
  for raw_cfg in raw_cfgs:
    mock_sem.get_framework_config.return_value = raw_cfg
    assert isinstance(HookContext(mock_sem, RuntimeConfig(target_framework="jax")).plugin_traits, PluginTraits)

  # `current_variant`, `lookup_api`, and `lookup_signature`
  assert ctx.current_variant is None
  ctx_with_op = HookContext(sem, RuntimeConfig(target_framework="jax"))
  ctx_with_op.current_op_id = "add"
  resolved_var = ctx_with_op.current_variant
  assert resolved_var is not None
  assert resolved_var.api == "jax.numpy.add"
  assert resolved_var.pack_to_tuple == "axes"

  ctx_missing_op = HookContext(sem, RuntimeConfig(target_framework="jax"))
  ctx_missing_op.current_op_id = "missing"
  assert ctx_missing_op.current_variant is None
  assert no_sym_ctx.current_variant is None

  direct_variant = FrameworkVariant(api="jax.numpy.subtract")
  mock_sem.resolve_variant.return_value = direct_variant
  direct_ctx = HookContext(mock_sem, RuntimeConfig(target_framework="jax"))
  direct_ctx.current_op_id = "sub"
  assert direct_ctx.current_variant is direct_variant

  mock_sem.resolve_variant.return_value = 123
  invalid_ctx = HookContext(mock_sem, RuntimeConfig(target_framework="jax"))
  invalid_ctx.current_op_id = "sub"
  assert invalid_ctx.current_variant is None
  assert invalid_ctx.lookup_api("sub") is None

  mock_sem.resolve_variant.return_value = {"api": 999}
  assert invalid_ctx.lookup_api("sub") is None

  assert ctx.lookup_api("add") == "jax.numpy.add"
  assert HookContext(sem, RuntimeConfig(target_framework="numpy")).lookup_api("add") == "numpy.add"
  assert HookContext(sem, RuntimeConfig(target_framework="tensorflow")).lookup_api("add") is None
  assert ctx.lookup_api("unknown_op") is None
  assert ctx.lookup_api("complex") is None
  assert no_sym_ctx.lookup_api("add") is None

  assert ctx.lookup_signature("add") == ["x1", "x2"]
  assert ctx.lookup_signature("abs") == ["x", "out", "plain_arg"]
  assert ctx.lookup_signature("bad_sig") == []
  assert ctx.lookup_signature("unknown_op") == []
  assert no_sym_ctx.lookup_signature("add") == []


def test_hook_registry_lifecycle_autowire_and_loader(tmp_path: Path) -> None:
  """Verify `HookRegistry`, `register_hook`, `bind_hook_registry`, and `_import_from_dir` discovery.

  Args:
      tmp_path: Pytest temporary directory fixture.
  """
  reg = get_default_hook_registry()
  clear_hooks(mark_loaded=True)
  assert len(reg.hooks) == 0
  assert get_hook("unknown_magic") is None

  @register_hook("sig_test")
  def _rename_call(node: cst.Call, _ctx: HookContext) -> cst.Call:
    """Rewrite call callee to `visited`.

    Args:
        node: Input `cst.Call`.
        _ctx: Active `HookContext`.

    Returns:
        Updated `cst.Call`.
    """
    return node.with_changes(func=cst.Name("visited"))

  hook = get_hook("sig_test")
  assert hook is not None
  res_node = hook(cst.Call(func=cst.Name("orig")), HookContext(None, RuntimeConfig()))
  assert isinstance(res_node, cst.Call)
  assert cst.ensure_type(res_node.func, cst.Name).value == "visited"

  # Overwriting an existing hook trigger replaces the handler
  @register_hook("conflict")
  def _hook_a(node: cst.CSTNode, _ctx: HookContext) -> cst.CSTNode:
    """First conflict handler.

    Args:
        node: Input CST node.
        _ctx: Active `HookContext`.

    Returns:
        `cst.Name('A')`.
    """
    _ = node
    return cst.Name("A")

  @register_hook("conflict")
  def _hook_b(node: cst.CSTNode, _ctx: HookContext) -> cst.CSTNode:
    """Replacement conflict handler.

    Args:
        node: Input CST node.
        _ctx: Active `HookContext`.

    Returns:
        `cst.Name('B')`.
    """
    _ = node
    return cst.Name("B")

  conflict_fn = get_hook("conflict")
  assert conflict_fn is not None
  assert cst.ensure_type(conflict_fn(cst.Name("x"), HookContext(None, RuntimeConfig())), cst.Name).value == "B"

  # Non-callable guard
  with pytest.raises(PluginLoadError, match="must be callable"):
    register_hook("bad_trigger")(123)

  isolated = HookRegistry(plugins_loaded=True)
  with pytest.raises(PluginLoadError, match="must be callable"):
    isolated.register("bad_isolated")(None)

  set_plugins_loaded(True)
  assert get_default_hook_registry().plugins_loaded is True
  set_plugins_loaded(False)
  assert get_default_hook_registry().plugins_loaded is False

  # Auto-wire metadata and isolated registry execution via ASTEngine
  clear_hooks(mark_loaded=True)

  @register_hook("auto_hook", auto_wire={"operation": "Op", "description": "doc", "variants": {}})
  def _auto(node: cst.CSTNode, _ctx: HookContext) -> cst.CSTNode:
    """Auto-wired hook stub.

    Args:
        node: Input CST node.
        _ctx: Active `HookContext`.

    Returns:
        Unchanged `node`.
    """
    return node

  assert "auto_hook" in get_all_hook_metadata()

  @isolated.register("isolated_op", auto_wire={"ops": {"CustomOp": {}}})
  def _isolated_hook(_node: cst.Call, _ctx: HookContext) -> cst.CSTNode:
    """Replace calls with `isolated_ok`.

    Args:
        _node: Input `cst.Call`.
        _ctx: Active `HookContext`.

    Returns:
        `cst.Name('isolated_ok')`.
    """
    return cst.Name("isolated_ok")

  assert HookContext(None, RuntimeConfig(), hook_registry=isolated).hook_registry is isolated
  with bind_hook_registry(isolated):
    assert get_hook("isolated_op") is _isolated_hook

  sem = SemanticsManager(load_defaults=False)
  raw_custom: dict[str, JsonValue] = {
    "operation": "CustomOp",
    "std_args": ["x"],
    "variants": {
      "torch": {"api": "torch.custom_fn"},
      "jax": {"api": "jax.custom_fn", "requires_plugin": "isolated_op"},
    },
  }
  sem.data["CustomOp"] = raw_custom
  sem.operations["CustomOp"] = OperationDef.model_validate(raw_custom)
  sem._build_index()

  engine_res = ASTEngine(
    semantics=sem,
    config=RuntimeConfig(source_framework="torch", target_framework="jax"),
    hook_registry=isolated,
  ).run(
    textwrap.dedent(
      """
      import torch
      y = torch.custom_fn(x)
      """
    )
  )
  assert engine_res.success is True
  assert "isolated_ok" in engine_res.code

  # Lazy load, custom plugin directory, missing directories, syntax errors, null spec, and reload
  clear_hooks(mark_loaded=False)
  with patch.object(reg, "load_plugins", return_value=0) as mock_load:
    assert get_hook("nonexistent") is None
    mock_load.assert_called_once()

  plugin_dir = tmp_path / "custom_plugins"
  plugin_dir.mkdir()
  (plugin_dir / "my_plugin.py").write_text(
    textwrap.dedent(
      """
      from ml_switcheroo.core.hooks import register_hook

      @register_hook('custom_trigger')
      def my_hook(node, ctx):
        return node
      """
    ),
    encoding="utf-8",
  )
  clear_hooks(mark_loaded=False)
  assert load_plugins(extra_dirs=[plugin_dir]) > 0
  assert "custom_trigger" in reg.hooks
  assert get_hook("inject_prng") is not None

  assert load_plugins(plugins_dir=tmp_path / "missing", extra_dirs=[tmp_path / "missing2"]) == 0

  bad_dir = tmp_path / "bad"
  bad_dir.mkdir()
  (bad_dir / "bad_mod.py").write_text("invalid python [", encoding="utf-8")
  with patch("logging.Logger.warning") as mock_warn:
    assert _import_from_dir(bad_dir, base_package=None) == 0
    mock_warn.assert_called_once()

  spec_dir = tmp_path / "null_spec"
  spec_dir.mkdir()
  (spec_dir / "plugin_mod.py").write_text("x = 1", encoding="utf-8")
  with (
    patch("importlib.util.spec_from_file_location", return_value=None),
    patch("logging.Logger.warning") as mock_warn_spec,
  ):
    assert _import_from_dir(spec_dir, base_package=None) == 0
    mock_warn_spec.assert_called_once()

  mod_dir = tmp_path / "mods"
  mod_dir.mkdir()
  (mod_dir / "good_mod.py").write_text("x = 1", encoding="utf-8")
  assert _import_from_dir(mod_dir, base_package=None) == 1
  assert _import_from_dir(mod_dir, base_package=None) == 1

  pkg_parent = tmp_path / "pkg_root"
  pkg_dir = pkg_parent / "temp_hook_pkg"
  pkg_dir.mkdir(parents=True)
  (pkg_dir / "__init__.py").write_text("", encoding="utf-8")
  (pkg_dir / "sample_hook_mod.py").write_text("VALUE = 1", encoding="utf-8")
  sys.path.insert(0, str(pkg_parent))
  try:
    assert _import_from_dir(pkg_dir, base_package="temp_hook_pkg") == 1
    assert _import_from_dir(pkg_dir, base_package="temp_hook_pkg") == 1
  finally:
    sys.path.remove(str(pkg_parent))
    sys.modules.pop("temp_hook_pkg.sample_hook_mod", None)
    sys.modules.pop("temp_hook_pkg", None)
