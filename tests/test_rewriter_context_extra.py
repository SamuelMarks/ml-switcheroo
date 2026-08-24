"""Test module."""

from ml_switcheroo.core.rewriter.context import RewriterContext
from unittest.mock import MagicMock
from ml_switcheroo.config import RuntimeConfig


def test_hydrate_alias_map_with_dict():
  """Test element."""
  # Hit the line 160->exit path
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  sm = MagicMock()
  sm.get_framework_config.return_value = {"alias": {"name": "th"}}
  ctx = RewriterContext(semantics=sm, config=config)

  assert ctx.alias_map.get("th") == "th"


def test_hydrate_alias_map_no_name():
  """Test element."""
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  sm = MagicMock()
  sm.get_framework_config.return_value = {"alias": {}}
  ctx = RewriterContext(semantics=sm, config=config)
  assert "th" not in ctx.alias_map


def test_hydrate_alias_map_model_dump():
  """Test element."""
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  sm = MagicMock()

  class DummyAlias:
    def model_dump(self):
      return {"name": "th_dummy"}

  sm.get_framework_config.return_value = {"alias": DummyAlias()}
  ctx = RewriterContext(semantics=sm, config=config)
  assert ctx.alias_map.get("th_dummy") == "th_dummy"


def test_hydrate_alias_map_dict_no_name_key():
  """Test element."""
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  sm = MagicMock()
  sm.get_framework_config.return_value = {"alias": {"something_else": "value"}}
  ctx = RewriterContext(semantics=sm, config=config)
  assert not ctx.alias_map


def test_hydrate_alias_map_not_dict():
  """Test element."""
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  sm = MagicMock()
  sm.get_framework_config.return_value = {"alias": "just_a_string"}
  ctx = RewriterContext(semantics=sm, config=config)
  assert not ctx.alias_map
