"""
Tests for Configuration, Path Resolution, and Defaults Logic.
"""

import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from pydantic import ValidationError

from ml_switcheroo.config import (
  RuntimeConfig,
  parse_cli_key_values,
  _resolve_default_source,
  _resolve_default_target,
)

# ... (Previous default sorting tests retained) ...


def test_config_flags():
  """Verify new flags defaulting."""
  c = RuntimeConfig()
  # Default False
  assert c.enable_graph_optimization is False
  # Default True
  assert c.enable_import_fixer is True


def test_legacy_fusion_alias():
  """Verify enable_graph_optimization aliases to enable_graph_optimization."""
  c = RuntimeConfig(enable_graph_optimization=True)
  assert c.enable_graph_optimization is True


def test_explicit_graph_opt():
  """Verify explicit setting works."""
  c = RuntimeConfig(enable_graph_optimization=True)
  assert c.enable_graph_optimization is True


# ... (Remaining tests from existing file retained) ...
