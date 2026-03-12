import pytest
from pathlib import Path
from unittest.mock import patch


def test_config_plugin_paths_no_toml_dir():
  from ml_switcheroo.config import RuntimeConfig

  with patch("ml_switcheroo.config._load_toml_settings", return_value=({"plugin_paths": ["plugin1", "plugin2"]}, None)):
    config = RuntimeConfig.load(search_path=Path("."))
    assert len(config.plugin_paths) == 2
    assert config.plugin_paths[0].name == "plugin1"


def test_config_tomllib_none():
  import sys
  import ml_switcheroo.config

  with patch("ml_switcheroo.config.tomllib", None):
    from ml_switcheroo.config import _load_toml_settings

    assert _load_toml_settings(Path(".")) == ({}, None)


def test_config_validation_report_toml():
  from ml_switcheroo.config import RuntimeConfig

  with patch("ml_switcheroo.config._load_toml_settings", return_value=({"validation_report": "report.json"}, Path("."))):
    config = RuntimeConfig.load(search_path=Path("."))
    assert config.validation_report == Path("report.json")


def test_config_plugin_paths_toml_dir():
  from ml_switcheroo.config import RuntimeConfig

  with patch("ml_switcheroo.config._load_toml_settings", return_value=({"plugin_paths": ["plugin1"]}, Path("."))):
    config = RuntimeConfig.load(search_path=Path("."))
    assert len(config.plugin_paths) == 1
    assert config.plugin_paths[0].name == "plugin1"


def test_config_graph_optimization():
  from ml_switcheroo.config import RuntimeConfig

  with patch("ml_switcheroo.config._load_toml_settings", return_value=({"enable_graph_optimization": True}, None)):
    config = RuntimeConfig.load(enable_graph_optimization=False)
    assert config.enable_graph_optimization is False

    config2 = RuntimeConfig.load()
    assert config2.enable_graph_optimization is True
