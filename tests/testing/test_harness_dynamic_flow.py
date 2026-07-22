"""Test suite for the Harness Dynamic Flow module."""

import pytest
from unittest.mock import patch
from ml_switcheroo.testing.harness_generator import HarnessGenerator


@pytest.fixture
def generator():
  """Provides a mock generator for testing."""
  return HarnessGenerator()


class MockAdapterWithMagic:
  """Mock Adapter With Magic class for testing purposes."""

  declared_magic_args = ["magic_k", "other_k"]
  harness_imports = ["import magic_lib"]

  def get_harness_init_code(self):
    """Mock implementation of get harness initialization code."""
    return "def _magic_helper(seed): return 'magic_val'"

  def convert(self, x):
    """Mock implementation of convert."""
    return x


class MockAdapterNoMagic:
  """Mock Adapter No Magic class for testing purposes."""

  declared_magic_args = []
  harness_imports = []

  def get_harness_init_code(self):
    """Mock implementation of get harness initialization code."""
    return ""

  def convert(self, x):
    """Mock implementation of convert."""
    return x


def test_dynamic_logic_injection(generator, tmp_path):
  """Verifies the behavior of dynamic logic injection."""
  target_key = "magic_fw"
  with patch("ml_switcheroo.testing.harness_generator.get_adapter") as mock_get:
    mock_get.return_value = MockAdapterWithMagic()
    out_file = tmp_path / "magic_verify.py"
    generator.generate(tmp_path, tmp_path, out_file, target_fw=target_key)
    content = out_file.read_text(encoding="utf-8")
    assert "import magic_lib" in content
    assert "def _magic_helper(seed):" in content
    assert 'if tp in ["magic_k", "other_k"]:' in content
    assert "val = _magic_helper(seed=42)" in content


def test_dynamic_logic_noop(generator, tmp_path):
  """Verifies the behavior of dynamic logic noop."""
  target_key = "plain_fw"
  with patch("ml_switcheroo.testing.harness_generator.get_adapter") as mock_get:
    mock_get.return_value = MockAdapterNoMagic()
    out_file = tmp_path / "plain_verify.py"
    generator.generate(tmp_path, tmp_path, out_file, target_fw=target_key)
    content = out_file.read_text(encoding="utf-8")
    assert "import magic_lib" not in content
    assert "val = None(seed=42)" not in content.replace("_", "")
    assert "# --- HELPERS FOR STATE INJECTION ---" in content
