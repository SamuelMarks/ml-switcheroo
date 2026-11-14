"""Test suite for the Physical Gen module."""

from pathlib import Path

import pytest
from pydantic import JsonValue

from ml_switcheroo.enums import RuntimeTensorOrValue
from ml_switcheroo.generated_tests.generator import SemanticsInputMap, TestCaseGenerator
from ml_switcheroo.semantics.manager import SemanticsManager


class MockSemantics(SemanticsManager):
  """Test double `MockSemantics` used in physical gen tests."""

  def __init__(self) -> None:
    """Initializes the MockSemantics instance."""
    super().__init__(load_defaults=False)
    self.data = {}
    self.test_templates: dict[str, dict[str, str]] = {
      "jax": {
        "import": "import jax\nimport jax.numpy as jnp",
        "convert_input": "jnp.array({np_var})",
        "to_numpy": "{res_var}",
        "jit_template": "jax.jit({fn}, static_argnums={static_argnums})",
      },
      "torch": {"import": "import torch", "convert_input": "torch.tensor({np_var})", "to_numpy": "{res_var}.numpy()"},
    }
    self.framework_configs = {"jax": {"traits": {"jit_static_args": ["axis", "keepdims"]}}}

  def get_test_template(self, fw: str) -> dict[str, str] | None:
    """Execute `get` to support get test template in physical gen tests.

    Args:
        fw: Framework identifier string (e.g., torch, jax, mlx).

    Returns:
        Resulting `dict[str, str] | None` from get test template.
    """
    return self.test_templates.get(fw)

  def get_framework_config(self, fw: str) -> dict[str, JsonValue]:
    """Mock implementation of get framework configuration.

    Args:
        fw: Framework identifier string (e.g., torch, jax, mlx).

    Returns:
        Resulting `dict[str, JsonValue]` from get framework config.
    """
    return self.framework_configs.get(fw, {})


@pytest.fixture
def generator() -> TestCaseGenerator:
  """Create a configured `TestCaseGenerator` fixture for physical gen tests.

  Returns:
      Configured `generator` fixture value (TestCaseGenerator).
  """
  mgr = MockSemantics()
  return TestCaseGenerator(semantics_mgr=mgr)


@pytest.fixture
def sample_spec() -> SemanticsInputMap:
  """Create a configured `SemanticsInputMap` fixture for physical gen tests.

  Returns:
      Configured `sample_spec` fixture value.
  """
  spec: dict[str, dict[str, RuntimeTensorOrValue]] = {
    "abs": {"std_args": ["x"], "variants": {"torch": {"api": "torch.abs"}, "jax": {"api": "jnp.abs"}}},
    "mean": {
      "std_args": [
        ("x", "Array"),
        ("axis", "int"),
      ],
      "variants": {"torch": {"api": "torch.mean", "args": {"axis": "dim"}}, "jax": {"api": "jnp.mean"}},
    },
  }
  return spec


def test_generator_writes_file_and_runtime(
  generator: TestCaseGenerator, sample_spec: SemanticsInputMap, tmp_path: Path
) -> None:
  """Verify `generator.generate` handles generator writes file and runtime as expected.

  Args:
      generator: Input `generator` parameter value (`TestCaseGenerator`).
      sample_spec: Input `sample_spec` parameter value (`SemanticsInputMap`).
      tmp_path: Pytest temporary directory path fixture.
  """
  out_file: Path = tmp_path / "test_generated.py"
  generator.generate(sample_spec, out_file)
  assert out_file.exists()
  content: str = out_file.read_text()
  assert "from .runtime import *" in content
  assert "verify_results(ref, val" in content
  runtime_file: Path = tmp_path / "runtime.py"
  assert runtime_file.exists()
  runtime_content: str = runtime_file.read_text()
  assert 'find_spec("torch")' in runtime_content
  assert "TORCH_AVAILABLE" in runtime_content
  assert 'find_spec("jax")' in runtime_content
  assert "chex_mod.assert_trees_all_close" in runtime_content
  assert "isinstance(ref, dict)" in runtime_content


def test_jit_static_argnums(generator: TestCaseGenerator, sample_spec: SemanticsInputMap, tmp_path: Path) -> None:
  """Verify `generator.generate` emits `static_argnums=(1,)` for JIT static argnums.

  Args:
      generator: Input `generator` parameter value (`TestCaseGenerator`).
      sample_spec: Input `sample_spec` parameter value (`SemanticsInputMap`).
      tmp_path: Pytest temporary directory path fixture.
  """
  out_file: Path = tmp_path / "test_jit.py"
  generator.generate(sample_spec, out_file)
  content: str = out_file.read_text()
  start: int = content.find("test_gen_mean")
  block: str = content[start:]
  assert "static_argnums=(1,)" in block
  assert "jax.jit(fn, static_argnums=(1,))" in block


def test_generated_file_is_valid_python(
  generator: TestCaseGenerator, sample_spec: SemanticsInputMap, tmp_path: Path
) -> None:
  """Verify `generator.generate` handles generated file is valid python as expected.

  Args:
      generator: Input `generator` parameter value (`TestCaseGenerator`).
      sample_spec: Input `sample_spec` parameter value (`SemanticsInputMap`).
      tmp_path: Pytest temporary directory path fixture.
  """
  import ast

  out_file: Path = tmp_path / "test_valid.py"
  generator.generate(sample_spec, out_file)
  try:
    code_obj = ast.parse(out_file.read_text(), out_file.name, "exec")
  except SyntaxError as e:
    pytest.fail(f"Generated Invalid Python: {e}")
  assert isinstance(code_obj, ast.Module)
  func_names: list[str] = [node.name for node in code_obj.body if isinstance(node, ast.FunctionDef)]
  assert "test_gen_abs" in func_names
  assert "test_gen_mean" in func_names


def test_overwrite_behavior(generator: TestCaseGenerator, sample_spec: SemanticsInputMap, tmp_path: Path) -> None:
  """Verify `generator.generate` emits `def test_gen_abs` for overwrite behavior.

  Args:
      generator: Input `generator` parameter value (`TestCaseGenerator`).
      sample_spec: Input `sample_spec` parameter value (`SemanticsInputMap`).
      tmp_path: Pytest temporary directory path fixture.
  """
  out_file: Path = tmp_path / "test_overwrite.py"
  out_file.write_text("OLD CONTENT")
  generator.generate(sample_spec, out_file)
  content: str = out_file.read_text()
  assert "OLD CONTENT" not in content
  assert "def test_gen_abs" in content


def test_skip_existing_manual_test(generator: TestCaseGenerator, sample_spec: SemanticsInputMap, tmp_path: Path) -> None:
  """Verify `generator.generate` emits `def test_gen_mean` for skip existing manual test.

  Args:
      generator: Input `generator` parameter value (`TestCaseGenerator`).
      sample_spec: Input `sample_spec` parameter value (`SemanticsInputMap`).
      tmp_path: Pytest temporary directory path fixture.
  """
  out_file: Path = tmp_path / "test_manual.py"
  out_file.write_text("def test_gen_abs(): pass")
  generator.generate(sample_spec, out_file)
  content: str = out_file.read_text()
  assert "def test_gen_abs" not in content
  assert "def test_gen_mean" in content


def test_empty_semantics(generator: TestCaseGenerator, tmp_path: Path) -> None:
  """Verifies empty semantics early exit and non-mapping entry skipping.

  Args:
      generator: Input `generator` parameter value (`TestCaseGenerator`).
      tmp_path: Pytest temporary directory path fixture.
  """
  out_file: Path = tmp_path / "test_empty.py"
  generator.generate({}, out_file)
  assert not out_file.exists()

  non_map_file: Path = tmp_path / "test_non_map.py"
  generator.generate({"invalid_op": "not_a_mapping"}, non_map_file)
  assert "insufficient variants" in non_map_file.read_text(encoding="utf-8")
