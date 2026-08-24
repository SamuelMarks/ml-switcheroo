"""Test extra dependencies."""

import sys
from unittest.mock import MagicMock
from ml_switcheroo.analysis.dependencies import DependencyScanner


def test_is_stdlib_py310(monkeypatch):
  """Test element."""
  monkeypatch.setattr(sys, "version_info", (3, 10))
  # We must patch sys.stdlib_module_names
  monkeypatch.setattr(sys, "stdlib_module_names", {"os", "sys"}, raising=False)

  scanner = DependencyScanner(MagicMock(), "torch")
  assert scanner._is_stdlib("os") is True
  assert scanner._is_stdlib("torch") is False
