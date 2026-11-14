"""Test suite for the Generator Template module."""

from pathlib import Path

import pytest
from pydantic import JsonValue

from ml_switcheroo.generated_tests.generator import TestCaseGenerator
from ml_switcheroo.semantics.manager import SemanticsManager


class MockTemplateSemantics(SemanticsManager):
  """Test double `MockTemplateSemantics` used in generator template tests."""

  def __init__(self, templates: dict[str, dict[str, str]] | None = None) -> None:
    """Initializes the MockTemplateSemantics instance.

    Args:
        templates: Framework template mapping definitions (`dict[str, dict[str, str]] | None`).
    """
    super().__init__(load_defaults=False)
    self.test_templates = templates or {}
    self.data = {}
    self.framework_configs = {}

  def get_test_template(self, framework: str) -> dict[str, str] | None:
    """Execute `get` to support get test template in generator template tests.

    Args:
        framework: Target or source framework identifier (`str`).

    Returns:
        Resulting `dict[str, str] | None` from get test template.
    """
    return self.test_templates.get(framework)

  def get_framework_config(self, framework: str) -> dict[str, JsonValue]:
    """Mock implementation of get framework configuration.

    Args:
        framework: Framework identifier string.

    Returns:
        Resulting `dict[str, JsonValue]` from get framework config.
    """
    return {}


@pytest.fixture
def semantics_data() -> dict[str, JsonValue]:
  """Create a configured `dict[str, JsonValue]` fixture for generator template tests.

  Returns:
      Configured `semantics_data` fixture value (dict[str, JsonValue]).
  """
  return {"abs": {"std_args": ["x"], "variants": {"torch": {"api": "torch.abs"}, "tinygrad": {"api": "tinygrad.abs"}}}}


def test_default_template_fallback(tmp_path: Path, semantics_data: dict[str, JsonValue]) -> None:
  """Verify `TestCaseGenerator` handles default template fallback as expected.

  Args:
      tmp_path: Pytest temporary directory path fixture.
      semantics_data: Input `semantics_data` parameter value (`dict[str, JsonValue]`).
  """
  defaults: dict[str, dict[str, str]] = {
    "torch": {"import": "import torch", "convert_input": "torch.tensor({np_var})", "to_numpy": "{res_var}.numpy()"},
    "jax": {"import": "import jax", "convert_input": "jnp.array({np_var})", "to_numpy": "{res_var}"},
  }
  mgr = MockTemplateSemantics(templates=defaults)
  gen = TestCaseGenerator(semantics_mgr=mgr)
  out_file: Path = tmp_path / "test_defaults.py"
  abs_entry = semantics_data["abs"]
  assert isinstance(abs_entry, dict)
  variants = abs_entry["variants"]
  assert isinstance(variants, dict)
  variants["jax"] = {"api": "jnp.abs"}
  gen.generate(semantics_data, out_file)
  content: str = out_file.read_text()
  runtime_content: str = (out_file.parent / "runtime.py").read_text()
  assert 'find_spec("torch")' in runtime_content
  assert 'find_spec("jax")' in runtime_content
  assert "Framework: tinygrad" not in content


def test_custom_backend_template(tmp_path: Path, semantics_data: dict[str, JsonValue]) -> None:
  """Verify `TestCaseGenerator` emits `Framework: tinygrad` for custom backend template.

  Args:
      tmp_path: Pytest temporary directory path fixture.
      semantics_data: Input `semantics_data` parameter value (`dict[str, JsonValue]`).
  """
  custom_templates: dict[str, dict[str, str]] = {
    "torch": {"import": "import torch", "convert_input": "torch.tensor({np_var})", "to_numpy": "{res_var}.numpy()"},
    "tinygrad": {
      "import": "from tinygrad.tensor import Tensor",
      "convert_input": "Tensor({np_var})",
      "to_numpy": "{res_var}.numpy()",
    },
  }
  mgr = MockTemplateSemantics(templates=custom_templates)
  gen = TestCaseGenerator(semantics_mgr=mgr)
  out_file: Path = tmp_path / "test_tinygrad.py"
  gen.generate(semantics_data, out_file)
  content: str = out_file.read_text()
  runtime_content: str = (out_file.parent / "runtime.py").read_text()
  assert "Framework: tinygrad" in content
  assert 'find_spec("tinygrad")' in runtime_content
  assert "Tensor(np_x)" in content


def test_jit_config_via_template(tmp_path: Path) -> None:
  """Verify `TestCaseGenerator` emits `Framework: custom_jit` for JIT config via template.

  Args:
      tmp_path: Pytest temporary directory path fixture.
  """
  data = {"add": {"std_args": ["x"], "variants": {"custom_jit": {"api": "lib.add"}, "other": {"api": "other.add"}}}}
  templates: dict[str, dict[str, str]] = {
    "custom_jit": {
      "import": "import lib",
      "convert_input": "{np_var}",
      "to_numpy": "{res_var}",
      "jit_template": "jax.jit({fn})",
    },
    "other": {"import": "import other", "convert_input": "{np_var}", "to_numpy": "{res_var}"},
  }
  mgr = MockTemplateSemantics(templates=templates)
  gen = TestCaseGenerator(semantics_mgr=mgr)
  out_file: Path = tmp_path / "test_jit_flag.py"
  gen.generate(data, out_file)
  content: str = out_file.read_text()
  assert "Framework: custom_jit" in content
  assert "jax.jit(fn)" in content


def test_invalid_framework_skipped(tmp_path: Path, semantics_data: dict[str, JsonValue]) -> None:
  """Verify `TestCaseGenerator` emits `Framework: jax` for invalid framework skipped.

  Args:
      tmp_path: Pytest temporary directory path fixture.
      semantics_data: Input `semantics_data` parameter value (`dict[str, JsonValue]`).
  """
  abs_entry = semantics_data["abs"]
  assert isinstance(abs_entry, dict)
  variants = abs_entry["variants"]
  assert isinstance(variants, dict)
  variants["ghost_fw"] = {"api": "ghost.abs"}
  variants["jax"] = {"api": "jnp.abs"}
  templates: dict[str, dict[str, str]] = {"torch": {"import": "import torch"}, "jax": {"import": "import jax"}}
  mgr = MockTemplateSemantics(templates=templates)
  gen = TestCaseGenerator(semantics_mgr=mgr)
  out_file: Path = tmp_path / "test_skip.py"
  gen.generate(semantics_data, out_file)
  content: str = out_file.read_text()
  assert "Framework: jax" in content
  assert "Framework: ghost_fw" not in content
