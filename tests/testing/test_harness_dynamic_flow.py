"""Test suite for the Harness Dynamic Flow module."""

from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

import pytest
from pydantic import JsonValue

from ml_switcheroo.testing.harness_generator import HarnessGenerator


@pytest.fixture
def generator() -> HarnessGenerator:
  """Create a configured `HarnessGenerator` fixture for harness dynamic flow tests.

  Returns:
      Configured `generator` fixture value (HarnessGenerator).
  """
  return HarnessGenerator()


class MockAdapterWithMagic:
  """Test double `MockAdapterWithMagic` used in harness dynamic flow tests."""

  declared_magic_args: ClassVar[list[str]] = ["magic_k", "other_k"]
  harness_imports: ClassVar[list[str]] = ["import magic_lib"]

  def get_harness_init_code(self) -> str:
    """Mock implementation of get harness initialization code.

    Returns:
        Resulting `str` from get harness init code.
    """
    return "def _magic_helper(seed): return 'magic_val'"

  def convert(self, x: JsonValue) -> JsonValue:
    """Mock implementation of convert.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Transpiled source code string or rewritten CST module.
    """
    return x


class MockAdapterNoMagic:
  """Test double `MockAdapterNoMagic` used in harness dynamic flow tests."""

  declared_magic_args: ClassVar[list[str]] = []
  harness_imports: ClassVar[list[str]] = []

  def get_harness_init_code(self) -> str:
    """Mock implementation of get harness initialization code.

    Returns:
        Resulting `str` from get harness init code.
    """
    return ""

  def convert(self, x: JsonValue) -> JsonValue:
    """Mock implementation of convert.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Transpiled source code string or rewritten CST module.
    """
    return x


def test_dynamic_logic_injection(generator: HarnessGenerator, tmp_path: Path) -> None:
  """Verify `generator.generate` emits `import magic_lib` for dynamic logic injection.

  Args:
      generator: Code or MLIR generator instance under test.
      tmp_path: Pytest temporary directory path fixture.
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
  """Verify `generator.generate` handles dynamic logic noop as expected.

  Args:
      generator: Code or MLIR generator instance under test.
      tmp_path: Pytest temporary directory path fixture.
  """
  target_key: str = "plain_fw"
  with patch("ml_switcheroo.testing.harness_generator.get_adapter") as mock_get:
    mock_get.return_value = MockAdapterNoMagic()
    out_file: Path = tmp_path / "plain_verify.py"
    generator.generate(tmp_path, tmp_path, out_file, target_framework=target_key)
    content: str = out_file.read_text(encoding="utf-8")
    assert "import magic_lib" not in content
    assert "val = None(seed=42)" not in content.replace("_", "")
