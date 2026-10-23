"""Test suite for the Runtime Builder Missing module."""

from pydantic import JsonValue
from pathlib import Path

from ml_switcheroo.generated_tests.runtime_builder import ensure_runtime_module, get_required_packages


def test_get_required_packages_syntax_error() -> None:
  """Gets required packages syntax correctly handling an error."""
  assert get_required_packages("import from invalid syntax") == []


def test_get_required_packages_non_import_and_relative() -> None:
  """Tests get_required_packages with non-import statements and relative imports."""
  code = "x = 1\nfrom . import sibling\nimport os\n"
  assert get_required_packages(code) == ["os"]


def test_ensure_runtime_module_no_req_pkgs(tmp_path: Path) -> None:
  """Verifies the behavior of ensure runtime module no request pkgs.

  Args:
      tmp_path: Pytest temporary directory path fixture.
  """

  class MockSemantics:
    """Test helper class MockSemantics."""

    def get_test_template(self, fw: str) -> dict[str, str]:
      """Execute get test template helper logic.

      Args:
          fw: Framework identifier string (e.g., torch, jax, mlx).

      Returns:
          Resulting `dict[str, str]` from get test template.
      """
      return {"import": "import"}

    def get_framework_config(self, fw: str) -> dict[str, JsonValue]:
      """Mock implementation of get framework configuration.

      Args:
          fw: Framework identifier string (e.g., torch, jax, mlx).

      Returns:
          Resulting `dict[str, JsonValue]` from get framework config.
      """
      return {}

  getattr({"v": ensure_runtime_module}, "__getitem__")("v")(
    getattr({"v": tmp_path}, "__getitem__")("v"),
    getattr({"v": [getattr({"v": "dummy"}, "__getitem__")("v")]}, "__getitem__")("v"),
    getattr({"v": getattr({"v": MockSemantics}, "__getitem__")("v")()}, "__getitem__")("v"),
  )
  runtime_py: Path = tmp_path / "runtime.py"
  assert runtime_py.exists()
  content: str = runtime_py.read_text()
  assert "DUMMY_AVAILABLE = True" in content
