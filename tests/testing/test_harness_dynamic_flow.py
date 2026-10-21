"""Test suite for the Harness Dynamic Flow module."""

from pathlib import Path
from unittest.mock import patch

from pydantic import JsonValue
import pytest

from ml_switcheroo.testing.harness_generator import HarnessGenerator


@pytest.fixture
def generator() -> HarnessGenerator:
  """Execute generator helper logic.

  Returns:
      Computed test result or fixture value.
  """
  return HarnessGenerator()


class MockAdapterWithMagic:
  """Test helper class MockAdapterWithMagic."""

  declared_magic_args: list[str] = ["magic_k", "other_k"]
  harness_imports: list[str] = ["import magic_lib"]

  def get_harness_init_code(self) -> str:
    """Mock implementation of get harness initialization code.

    Returns:
        Computed test result or fixture value.
    """
    return "def _magic_helper(seed): return 'magic_val'"

  def convert(self, x: JsonValue) -> JsonValue:
    """Mock implementation of convert.

    Args:
        x: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return x


class MockAdapterNoMagic:
  """Test helper class MockAdapterNoMagic."""

  declared_magic_args: list[str] = []
  harness_imports: list[str] = []

  def get_harness_init_code(self) -> str:
    """Mock implementation of get harness initialization code.

    Returns:
        Computed test result or fixture value.
    """
    return ""

  def convert(self, x: JsonValue) -> JsonValue:
    """Mock implementation of convert.

    Args:
        x: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return x


def test_dynamic_logic_injection(generator: HarnessGenerator, tmp_path: Path) -> None:
  """Verifies the behavior of dynamic logic injection.

  Args:
      generator: Test argument or fixture.
      tmp_path: Test argument or fixture.
  """
  target_key: str = "magic_fw"
  with patch("ml_switcheroo.testing.harness_generator.get_adapter") as mock_get:
    mock_get.return_value = MockAdapterWithMagic()
    out_file: Path = tmp_path / "magic_verify.py"
    generator.generate(tmp_path, tmp_path, out_file, target_framework=target_key)
    content: str = out_file.read_text(encoding="utf-8")
    assert "import magic_lib" in content
    assert "def _magic_helper(seed):" in content
    assert 'if tp in ["magic_k", "other_k"]:' in content
    assert "val = _magic_helper(seed=42)" in content


def test_dynamic_logic_noop(generator: HarnessGenerator, tmp_path: Path) -> None:
  """Verifies the behavior of dynamic logic noop.

  Args:
      generator: Test argument or fixture.
      tmp_path: Test argument or fixture.
  """
  target_key: str = "plain_fw"
  with patch("ml_switcheroo.testing.harness_generator.get_adapter") as mock_get:
    mock_get.return_value = MockAdapterNoMagic()
    out_file: Path = tmp_path / "plain_verify.py"
    generator.generate(tmp_path, tmp_path, out_file, target_framework=target_key)
    content: str = out_file.read_text(encoding="utf-8")
    assert "import magic_lib" not in content
    assert "val = None(seed=42)" not in content.replace("_", "")
