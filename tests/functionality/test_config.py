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
