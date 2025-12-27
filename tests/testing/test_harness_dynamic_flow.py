"""
Tests for Dynamic Harness Generation Logic.

Verifies that:
1. Queries to `get_harness_init_code` determine the injected helper.
2. `declared_magic_args` determines the dispatch logic in the loop.
3. Imports are correctly placed.
"""

import pytest
from unittest.mock import MagicMock, patch
from ml_switcheroo.testing.harness_generator import HarnessGenerator


# --- Fixtures ---


@pytest.fixture
def generator():
  return HarnessGenerator()


class MockAdapterWithMagic:
  """
  Simulates a framework requiring init (like JAX).
  """

  declared_magic_args = ["magic_k", "other_k"]
  harness_imports = ["import magic_lib"]

  def get_harness_init_code(self):
    return "def _magic_helper(seed): return 'magic_val'"

    # Required stub

  def convert(self, x):
    return x


class MockAdapterNoMagic:
  """Simulates a framework like Torch."""

  declared_magic_args = []
  harness_imports = []

  def get_harness_init_code(self):
    return ""

  def convert(self, x):
    return x


# --- Tests ---


def test_dynamic_logic_injection(generator, tmp_path):
  """
  Scenario: Target framework has magic args (MockAdapterWithMagic).
  Expectation:
      1. 'import magic_lib' present.
      2. '_magic_helper' definition present.
      3. Dispatch logic: `if tp in ["magic_k", "other_k"]: val = _magic_helper(...)`
  """
  target_key = "magic_fw"

  with patch("ml_switcheroo.testing.harness_generator.get_adapter") as mock_get:
    mock_get.return_value = MockAdapterWithMagic()

    # Generate to string directly (or file then read)
    out_file = tmp_path / "magic_verify.py"

    # We don't care about source/target path content for this structure check
    generator.generate(tmp_path, tmp_path, out_file, target_fw=target_key)

    content = out_file.read_text(encoding="utf-8")

    # 1. Imports
    assert "import magic_lib" in content

    # 2. Helper def
    assert "def _magic_helper(seed):" in content

    # 3. Dispatch Logic
    # Check that the list string was formatted
    assert 'if tp in ["magic_k", "other_k"]:' in content
    # Check that it calls the extracted function name
    assert "val = _magic_helper(seed=42)" in content


def test_dynamic_logic_noop(generator, tmp_path):
  """
  Scenario: Target framework has no magic (MockAdapterNoMagic).
  Expectation: No imports, empty init, pass logic.
  """
  target_key = "plain_fw"

  with patch("ml_switcheroo.testing.harness_generator.get_adapter") as mock_get:
    mock_get.return_value = MockAdapterNoMagic()

    out_file = tmp_path / "plain_verify.py"
    generator.generate(tmp_path, tmp_path, out_file, target_fw=target_key)

    content = out_file.read_text(encoding="utf-8")

    # No magic imports
    assert "import magic_lib" not in content

    # Default pass logic inside injection block
    # Because we replace {param_injection_logic}, and _build returns "pass"
    # We look for "pass" specifically where logic would be.
    # But 'pass' is common.

    # Ensure regex extraction didn't inject a broken function call
    assert "val = None(seed=42)" not in content.replace("_", "")

    # Check standard sections existed
    assert "# --- HELPERS FOR STATE INJECTION ---" in content
