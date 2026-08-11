"""Test suite for the Config module."""

from ml_switcheroo.config import RuntimeConfig


def test_config_flags():
  """Verifies the behavior of configuration flags."""
  c = RuntimeConfig()
  assert c.enable_graph_optimization is False
  assert c.enable_sharding is False
  assert c.enable_import_fixer is True


def test_legacy_fusion_alias():
  """Verifies the behavior of legacy fusion alias."""
  c = RuntimeConfig(enable_graph_optimization=True)
  assert c.enable_graph_optimization is True


def test_explicit_graph_opt():
  """Verifies the behavior of explicit graph option."""
  c = RuntimeConfig(enable_graph_optimization=True)
  assert c.enable_graph_optimization is True


def test_config_ui_priority_invalid(monkeypatch):
  """Verifies the behavior when ui_priority is invalid."""
  from ml_switcheroo.config import get_framework_priority_order

  class MockAdapter:
    inherits_from = None
    ui_priority = "not_an_int"

  with monkeypatch.context() as m:
    m.setattr("ml_switcheroo.frameworks.base.get_adapter", lambda n: MockAdapter())
    m.setattr("ml_switcheroo.frameworks.base.available_frameworks", lambda: ["b", "a"])
    res = get_framework_priority_order()
    # It will fallback to priority 999 and sort alphabetically for a tie
    assert res == ["a", "b"]


def test_config_ui_priority_no_adapter(monkeypatch):
  """Test function."""
  from ml_switcheroo.config import get_framework_priority_order

  with monkeypatch.context() as m:
    m.setattr("ml_switcheroo.frameworks.base.get_adapter", lambda n: None)
    m.setattr("ml_switcheroo.frameworks.base.available_frameworks", lambda: ["b", "a"])
    res = get_framework_priority_order()
    assert res == ["a", "b"]


def test_config_ui_priority_has_adapter_but_no_ui_priority(monkeypatch):
  """Test function."""
  from ml_switcheroo.config import get_framework_priority_order

  class MockAdapter:
    inherits_from = None
    # no ui_priority

  with monkeypatch.context() as m:
    m.setattr("ml_switcheroo.frameworks.base.get_adapter", lambda n: MockAdapter())
    m.setattr("ml_switcheroo.frameworks.base.available_frameworks", lambda: ["a"])
    res = get_framework_priority_order()
    assert res == ["a"]


def test_config_ui_priority_has_adapter_with_value_error(monkeypatch):
  """Test function."""
  from ml_switcheroo.config import get_framework_priority_order

  class MockAdapter:
    inherits_from = None
    ui_priority = "not_an_int"

  with monkeypatch.context() as m:
    m.setattr("ml_switcheroo.frameworks.base.get_adapter", lambda n: MockAdapter())
    m.setattr("ml_switcheroo.frameworks.base.available_frameworks", lambda: ["a"])
    res = get_framework_priority_order()
    assert res == ["a"]


def test_config_ui_priority_has_adapter_with_type_error(monkeypatch):
  """Test function."""
  from ml_switcheroo.config import get_framework_priority_order

  class MockAdapter:
    inherits_from = None
    ui_priority = [1]  # list cannot be passed to int() natively, raises TypeError

  with monkeypatch.context() as m:
    m.setattr("ml_switcheroo.frameworks.base.get_adapter", lambda n: MockAdapter())
    m.setattr("ml_switcheroo.frameworks.base.available_frameworks", lambda: ["a"])
    res = get_framework_priority_order()
    assert res == ["a"]


def test_config_load_enable_sharding_none():
  """Test function."""
  from ml_switcheroo.config import RuntimeConfig
  from unittest.mock import patch

  with patch("ml_switcheroo.config._load_toml_settings", return_value=({"enable_sharding": True}, None)):
    # Testing when enable_sharding is explicitly None
    config = RuntimeConfig.load(enable_sharding=None)
    assert config.enable_sharding is True


def test_config_load_enable_sharding_not_none():
  """Test function."""
  from ml_switcheroo.config import RuntimeConfig
  from unittest.mock import patch

  with patch("ml_switcheroo.config._load_toml_settings", return_value=({"enable_sharding": False}, None)):
    # Testing when enable_sharding is explicitly True
    config = RuntimeConfig.load(enable_sharding=True)
    assert config.enable_sharding is True
