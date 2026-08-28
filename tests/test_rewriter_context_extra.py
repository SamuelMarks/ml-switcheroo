"""Test module."""

from ml_switcheroo.core.rewriter.context import RewriterContext
from unittest.mock import MagicMock
from ml_switcheroo.config import RuntimeConfig
from typing import Dict


def test_hydrate_alias_map_with_dict() -> None:
  """Test element."""
  # Hit the line 160->exit path
  config: RuntimeConfig = RuntimeConfig(source_framework="torch", target_framework="jax")
  sm: MagicMock = MagicMock()
  sm.get_framework_config.return_value = {"alias": {"name": "th"}}
  ctx: RewriterContext = RewriterContext(semantics=sm, config=config)

  assert ctx.alias_map.get("th") == "th"


def test_hydrate_alias_map_no_name() -> None:
  """Test element."""
  config: RuntimeConfig = RuntimeConfig(source_framework="torch", target_framework="jax")
  sm: MagicMock = MagicMock()
  sm.get_framework_config.return_value = {"alias": {}}
  ctx: RewriterContext = RewriterContext(semantics=sm, config=config)
  assert "th" not in ctx.alias_map


def test_hydrate_alias_map_model_dump() -> None:
  """Test element."""
  config: RuntimeConfig = RuntimeConfig(source_framework="torch", target_framework="jax")
  sm: MagicMock = MagicMock()

  class DummyAlias:
    def model_dump(self) -> Dict[str, str]:
      return {"name": "th_dummy"}

  sm.get_framework_config.return_value = {"alias": DummyAlias()}
  ctx: RewriterContext = RewriterContext(semantics=sm, config=config)
  assert ctx.alias_map.get("th_dummy") == "th_dummy"


def test_hydrate_alias_map_dict_no_name_key() -> None:
  """Test element."""
  config: RuntimeConfig = RuntimeConfig(source_framework="torch", target_framework="jax")
  sm: MagicMock = MagicMock()
  sm.get_framework_config.return_value = {"alias": {"something_else": "value"}}
  ctx: RewriterContext = RewriterContext(semantics=sm, config=config)
  assert not ctx.alias_map


def test_hydrate_alias_map_not_dict() -> None:
  """Test element."""
  config: RuntimeConfig = RuntimeConfig(source_framework="torch", target_framework="jax")
  sm: MagicMock = MagicMock()
  sm.get_framework_config.return_value = {"alias": "just_a_string"}
  ctx: RewriterContext = RewriterContext(semantics=sm, config=config)
  assert not ctx.alias_map
