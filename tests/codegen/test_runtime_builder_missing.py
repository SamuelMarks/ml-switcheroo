"""Test suite for the Runtime Builder Missing module."""

import typing
from pathlib import Path
from ml_switcheroo.generated_tests.runtime_builder import get_required_packages, ensure_runtime_module


def test_get_required_packages_syntax_error() -> None:
  """Gets required packages syntax correctly handling an error."""
  assert get_required_packages("import from invalid syntax") == []


def test_ensure_runtime_module_no_req_pkgs(tmp_path: Path) -> None:
  """Verifies the behavior of ensure runtime module no request pkgs."""

  class MockSemantics:
    """Mock Semantics class for testing purposes."""

    def get_test_template(self, fw: str) -> dict[str, str]:
      """Mock implementation of get test template."""
      return {"import": "import"}

    def get_framework_config(self, fw: str) -> dict[str, typing.Any]:
      """Mock implementation of get framework configuration."""
      return {}

  ensure_runtime_module(tmp_path, ["dummy"], MockSemantics())  # type: ignore
  runtime_py: Path = tmp_path / "runtime.py"
  assert runtime_py.exists()
  content: str = runtime_py.read_text()
  assert "DUMMY_AVAILABLE = True" in content
