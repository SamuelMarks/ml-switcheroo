"""Test suite for the Config Extra Py310 module."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from unittest import mock
from unittest.mock import patch

import pytest
from pydantic import BaseModel

import ml_switcheroo.config as conf
from ml_switcheroo.config import RuntimeConfig, get_framework_priority_order, parse_cli_key_values


def test_config_tomli_import() -> None:
  """Verifies the behavior of configuration tomli import."""
  from ml_switcheroo.config import _import_tomllib

  with mock.patch("sys.version_info", (3, 10, 0, "final", 0)):
    with mock.patch.dict("sys.modules", {"tomli": None, "tomllib": None}):
      assert _import_tomllib() is None
  with mock.patch("sys.version_info", (3, 10, 0, "final", 0)):
    mock_tomli: mock.MagicMock = mock.MagicMock()
    with mock.patch.dict("sys.modules", {"tomli": mock_tomli, "tomllib": None}):
      assert _import_tomllib() is mock_tomli

  with mock.patch("sys.version_info", (3, 11, 0, "final", 0)):
    mock_tomllib: mock.MagicMock = mock.MagicMock()
    with mock.patch.dict("sys.modules", {"tomllib": mock_tomllib}):
      assert _import_tomllib() is mock_tomllib


# --- Merged from test_config_extra.py ---


def test_config_tomli_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verifies the behavior of configuration tomli fallback."""
  monkeypatch.setattr(conf, "tomllib", None)
  assert conf.tomllib is None


def test_parse_cli_key_values() -> None:
  """Parses CLI key values."""
  assert parse_cli_key_values(None) == {}
  assert parse_cli_key_values([]) == {}
  items: List[str] = [
    "invalid_format",
    "key1=value1",
    "key_true=True",
    "key_false=fAlse",
    "key_int=42",
    "key_float=3.14",
    "key_exp=1e-5",
    "key_str_num=42a",
  ]
  res: Dict[str, Union[str, bool, int, float]] = parse_cli_key_values(items)
  assert res["key1"] == "value1"


def test_runtime_config_validate_frameworks() -> None:
  """Verifies the behavior of runtime configuration validate frameworks."""
  with pytest.raises(ValueError, match="Unknown framework"):
    RuntimeConfig(source_framework="unknown")


def test_get_plugin_config_validation() -> None:
  """Gets plugin configuration validation."""
  config: RuntimeConfig = RuntimeConfig(source_framework="jax", target_framework="torch", plugin_settings={"x": "bad"})

  class DummySchema(BaseModel):
    x: int

  with pytest.raises(ValueError):
    config.parse_plugin_settings(DummySchema)


def test_load_toml_config_missing() -> None:
  """Loads toml configuration missing."""
  res: Dict[str, Union[str, int, float, bool, list[str]]]
  p: Optional[Path]
  (res, p) = conf._load_toml_settings(start_path=conf.Path("/nonexistent_path_to_toml_dir"))
  assert res == {}
  assert p is None


def test_runtime_config_default_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verifies the behavior of runtime configuration default fallback."""
  import ml_switcheroo.frameworks.base as fb

  monkeypatch.setattr(fb, "available_frameworks", lambda: [])
  assert conf._resolve_default_target() == "target_placeholder"

  class BadAdapter:
    @property
    def ui_priority(self) -> str:
      """Helper to UI priority."""
      return "bad"

  monkeypatch.setattr(fb, "available_frameworks", lambda: ["dummy"])
  monkeypatch.setattr(fb, "get_adapter", lambda fw: BadAdapter())
  assert get_framework_priority_order() == ["dummy"]


def test_from_toml_path() -> None:
  """Verifies the behavior of from toml path."""
  import tempfile
  from pathlib import Path

  with tempfile.TemporaryDirectory() as d:
    p: Path = Path(d) / "pyproject.toml"
    with open(p, "w") as f:
      f.write(
        '[tool.ml_switcheroo]\nsource_framework = "jax"\ntarget_framework = "torch"\nenable_sharding = true\nvalidation_report = "custom_report.json"'
      )
    cfg: RuntimeConfig = RuntimeConfig.load(search_path=Path(d))
    assert cfg.source_framework == "jax"
    assert cfg.target_framework == "torch"
    assert cfg.enable_sharding is True
    assert str(cfg.validation_report) == "custom_report.json"
    with open(p, "w") as f:
      f.write("malformed [")
    RuntimeConfig.load(search_path=Path(d))


# --- Merged from test_config_extra2.py ---


def test_config_plugin_paths_no_toml_dir() -> None:
  """Verifies the behavior of configuration plugin paths no toml directory."""
  from ml_switcheroo.config import RuntimeConfig

  with patch("ml_switcheroo.config._load_toml_settings", return_value=({"plugin_paths": ["plugin1", "plugin2"]}, None)):
    config: RuntimeConfig = RuntimeConfig.load(search_path=Path("."))
    assert len(config.plugin_paths) == 2
    assert config.plugin_paths[0].name == "plugin1"


def test_config_tomllib_none() -> None:
  """Verifies the behavior of configuration tomllib none."""
  with patch("ml_switcheroo.config.tomllib", None):
    from ml_switcheroo.config import _load_toml_settings

    res: Tuple[Dict[str, Union[str, int, bool, list[str]]], Optional[Path]] = _load_toml_settings(Path("."))
    assert res == ({}, None)


def test_config_validation_report_toml() -> None:
  """Verifies the behavior of configuration validation report toml."""
  from ml_switcheroo.config import RuntimeConfig

  with patch("ml_switcheroo.config._load_toml_settings", return_value=({"validation_report": "report.json"}, Path("."))):
    config: RuntimeConfig = RuntimeConfig.load(search_path=Path("."))
    assert config.validation_report == Path("report.json")


def test_config_plugin_paths_toml_dir() -> None:
  """Verifies the behavior of configuration plugin paths toml directory."""
  from ml_switcheroo.config import RuntimeConfig

  with patch("ml_switcheroo.config._load_toml_settings", return_value=({"plugin_paths": ["plugin1"]}, Path("."))):
    config: RuntimeConfig = RuntimeConfig.load(search_path=Path("."))
    assert len(config.plugin_paths) == 1
    assert config.plugin_paths[0].name == "plugin1"


def test_config_graph_optimization() -> None:
  """Verifies the behavior of configuration graph optimization."""
  from ml_switcheroo.config import RuntimeConfig

  with patch("ml_switcheroo.config._load_toml_settings", return_value=({"enable_graph_optimization": True}, None)):
    config: RuntimeConfig = RuntimeConfig.load(enable_graph_optimization=False)
    assert config.enable_graph_optimization is False
    config2: RuntimeConfig = RuntimeConfig.load()
    assert config2.enable_graph_optimization is True
