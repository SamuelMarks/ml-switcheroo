"""Unit tests for `RuntimeConfig`, TOML discovery, and CLI flag parsing (`ml_switcheroo.config`)."""

import typing
from pathlib import Path
from unittest import mock

import pytest
from pydantic import BaseModel, Field

import ml_switcheroo.config as conf
import ml_switcheroo.frameworks.base as fb
from ml_switcheroo.config import (
  RuntimeConfig,
  RuntimeConfigOverrides,
  _import_tomllib,
  _load_toml_settings,
  _resolve_default_source,
  _resolve_default_target,
  get_framework_priority_order,
  parse_cli_key_values,
  parse_input_shapes_string,
  validate_intermediate_format,
)

_NL = chr(10)


class _SamplePluginSchema(BaseModel):
  """Pydantic schema used to test `RuntimeConfig.parse_plugin_settings`."""

  my_int: int = Field(...)


def test_runtime_config_defaults_flavours_and_plugin_validation() -> None:
  """Verify default flags, `effective_source`/`effective_target`, framework validation, and plugin schemas."""
  default_cfg = RuntimeConfig()
  assert default_cfg.enable_graph_optimization is False
  assert default_cfg.enable_sharding is False
  assert default_cfg.enable_import_fixer is True

  flavoured = RuntimeConfig(
    source_framework="torch",
    target_framework="jax",
    source_flavour="torchvision",
    target_flavour="flax_nnx",
    enable_graph_optimization=True,
  )
  assert flavoured.effective_source == "torchvision"
  assert flavoured.effective_target == "flax_nnx"
  assert flavoured.enable_graph_optimization is True

  plain = RuntimeConfig(source_framework="torch", target_framework="jax")
  assert plain.effective_source == "torch"
  assert plain.effective_target == "jax"

  with pytest.raises(ValueError, match="Unknown framework"):
    RuntimeConfig(source_framework="unknown_nonexistent_fw")

  valid_settings = RuntimeConfig(plugin_settings={"my_int": 5}).parse_plugin_settings(_SamplePluginSchema)
  assert valid_settings.my_int == 5

  with pytest.raises(ValueError, match="Plugin configuration validation failed"):
    RuntimeConfig(plugin_settings={"my_int": "bad"}).parse_plugin_settings(_SamplePluginSchema)

  assert validate_intermediate_format("ml_switcheroo_ir") == "ml_switcheroo_ir"
  assert parse_input_shapes_string("") is None
  assert parse_input_shapes_string("auto") is None
  assert parse_input_shapes_string("; () ; ( , )") is None
  assert parse_input_shapes_string("(2, bad)") is None
  assert parse_input_shapes_string("(2, 0)") is None
  assert parse_input_shapes_string("(2, , 4); [1, 8]") == [(2, 4), (1, 8)]


def test_parse_cli_key_values_and_default_resolvers(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `parse_cli_key_values` scalar coercion and default source/target resolution.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  assert parse_cli_key_values(None) == {}
  assert parse_cli_key_values([]) == {}
  parsed = parse_cli_key_values(
    [
      "invalid_without_equals",
      "k_str=v1",
      "k_true=true",
      "k_false=FALSE",
      "k_float=1.23",
      "k_int=42",
      "k_exp=1e-5",
      "k_alnum=42a",
    ]
  )
  assert parsed == {
    "k_str": "v1",
    "k_true": True,
    "k_false": False,
    "k_float": 1.23,
    "k_int": 42,
    "k_exp": 1e-5,
    "k_alnum": "42a",
  }

  monkeypatch.setattr(conf, "get_framework_priority_order", list)
  assert _resolve_default_source() == "source_placeholder"
  assert _resolve_default_target() == "target_placeholder"

  monkeypatch.setattr(conf, "get_framework_priority_order", lambda: ["only_fw"])
  assert _resolve_default_source() == "only_fw"
  assert _resolve_default_target() == "only_fw"


def test_framework_priority_ordering(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `get_framework_priority_order` handles missing adapters, child flavours, and invalid priorities.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """

  class _ChildAdapter:
    """Adapter inheriting from `parent`."""

    inherits_from = "parent"
    ui_priority = 10

  class _ParentAdapter:
    """Root adapter with valid priority."""

    inherits_from = None
    ui_priority = 20

  class _StrPriorityAdapter:
    """Adapter whose `ui_priority` raises `ValueError` in `int(...)`."""

    inherits_from = None
    ui_priority = "not_an_int"

  class _ListPriorityAdapter:
    """Adapter whose `ui_priority` raises `TypeError` in `int(...)`."""

    inherits_from = None
    ui_priority: typing.ClassVar[list[int]] = [1]

  class _NoPriorityAdapter:
    """Adapter without `ui_priority` attribute."""

    inherits_from = None

  class _BareAdapter:
    """Adapter lacking both `inherits_from` and `ui_priority`."""

  adapters: dict[
    str,
    _ChildAdapter | _ParentAdapter | _StrPriorityAdapter | _ListPriorityAdapter | _NoPriorityAdapter | _BareAdapter,
  ] = {
    "child": _ChildAdapter(),
    "parent": _ParentAdapter(),
    "bad_val": _StrPriorityAdapter(),
    "bad_type": _ListPriorityAdapter(),
    "noprio": _NoPriorityAdapter(),
    "bare": _BareAdapter(),
  }
  monkeypatch.setattr(
    fb,
    "available_frameworks",
    lambda: ["child", "bad_val", "parent", "no_adapter", "bad_type", "noprio", "bare"],
  )
  monkeypatch.setattr(fb, "get_adapter", adapters.get)
  order = get_framework_priority_order()
  assert order[0] == "parent"
  assert order[-1] == "child"
  assert set(order) == {"child", "bad_val", "parent", "no_adapter", "bad_type", "noprio", "bare"}


def test_toml_discovery_overrides_and_tomllib_import(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `RuntimeConfig.load` against real `tmp_path` `pyproject.toml` files and `_import_tomllib` fallbacks.

  Args:
      tmp_path: Pytest temporary directory fixture.
      monkeypatch: Pytest monkeypatch fixture.
  """
  # Missing pyproject.toml directory
  missing_dir = tmp_path / "missing_sub"
  assert _load_toml_settings(start_path=missing_dir) == ({}, None)

  # Valid pyproject.toml with relative `plugin_paths`, `validation_report`, `enable_sharding`, `enable_graph_optimization`
  proj_dir = tmp_path / "proj"
  nested_dir = proj_dir / "src" / "pkg"
  nested_dir.mkdir(parents=True)
  toml_path = proj_dir / "pyproject.toml"
  toml_path.write_text(
    _NL.join(
      [
        "[tool.ml_switcheroo]",
        'source_framework = "jax"',
        'target_framework = "torch"',
        "enable_sharding = true",
        "enable_graph_optimization = true",
        'validation_report = "custom_report.json"',
        'plugin_paths = ["plugins_a", "plugins_b"]',
      ]
    ),
    encoding="utf-8",
  )

  loaded = RuntimeConfig.load(search_path=nested_dir)
  assert loaded.source_framework == "jax"
  assert loaded.target_framework == "torch"
  assert loaded.enable_sharding is True
  assert loaded.enable_graph_optimization is True
  assert loaded.validation_report == Path("custom_report.json")
  assert [p.name for p in loaded.plugin_paths] == ["plugins_a", "plugins_b"]

  # Explicit CLI overrides take precedence over TOML booleans
  overridden = RuntimeConfig.load(
    RuntimeConfigOverrides(
      strict_mode=True,
      enable_graph_optimization=False,
      enable_sharding=False,
      search_path=proj_dir,
    )
  )
  assert overridden.strict_mode is True
  assert overridden.enable_graph_optimization is False
  assert overridden.enable_sharding is False

  # `plugin_paths` when `toml_dir` is None
  with mock.patch("ml_switcheroo.config._load_toml_settings", return_value=({"plugin_paths": ["rel_plug"]}, None)):
    no_dir_cfg = RuntimeConfig.load(search_path=proj_dir)
    assert [p.name for p in no_dir_cfg.plugin_paths] == ["rel_plug"]

  # Malformed TOML and `tomllib = None` fallback
  toml_path.write_text("malformed = [", encoding="utf-8")
  assert _load_toml_settings(proj_dir) == ({}, None)

  monkeypatch.setattr(conf, "tomllib", None)
  assert _load_toml_settings(proj_dir) == ({}, None)

  # `_import_tomllib` across Python >= 3.11 and Python 3.10 with/without `tomli`
  import sys

  fake_mod = mock.MagicMock()
  with pytest.MonkeyPatch.context() as mp:
    mp.setattr(sys, "version_info", (3, 11, 0, "final", 0))
    mp.setitem(sys.modules, "tomllib", fake_mod)
    assert _import_tomllib() is fake_mod

  with pytest.MonkeyPatch.context() as mp:
    mp.setattr(sys, "version_info", (3, 10, 0, "final", 0))
    mp.setitem(sys.modules, "tomli", fake_mod)
    mp.setitem(sys.modules, "tomllib", None)
    assert _import_tomllib() is fake_mod

  with pytest.MonkeyPatch.context() as mp:
    mp.setattr(sys, "version_info", (3, 10, 0, "final", 0))
    mp.setitem(sys.modules, "tomli", None)
    mp.setitem(sys.modules, "tomllib", None)
    assert _import_tomllib() is None
