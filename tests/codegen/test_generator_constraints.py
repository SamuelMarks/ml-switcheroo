"""Test suite for the Generator Constraints module."""

import pytest
from pathlib import Path
import typing
from unittest.mock import MagicMock
from ml_switcheroo.generated_tests.generator import TestCaseGenerator
from ml_switcheroo.semantics.manager import SemanticsManager


@pytest.fixture
def generator(tmp_path: Path) -> TestCaseGenerator:
  """Provides a mock generator for testing."""
  mgr: SemanticsManager = MagicMock(spec=SemanticsManager)
  templates: dict[str, dict[str, str]] = {
    "torch": {"import": "import torch", "convert_input": "{np_var}", "to_numpy": "{res_var}"}
  }
  mgr.get_test_template = MagicMock(side_effect=lambda fw: templates.get(fw))
  setattr(mgr, "test_templates", templates)
  mgr.get_framework_config = MagicMock(return_value={})
  return TestCaseGenerator(semantics_mgr=mgr)


def test_generate_options_constraint(generator: TestCaseGenerator, tmp_path: Path) -> None:
  """Generates options constraint."""
  semantics: dict[str, typing.Any] = {
    "opt_op": {
      "std_args": [{"name": "mode", "type": "str", "options": ["mean", "sum"]}],
      "variants": {"torch": {"api": "foo"}, "jax": {"api": "bar"}},
    }
  }
  out_file: Path = tmp_path / "test_opts.py"
  generator.generate(semantics, out_file)
  content: str = out_file.read_text()
  assert "random.choice(['mean', 'sum'])" in content


def test_generate_int_range_constraint(generator: TestCaseGenerator, tmp_path: Path) -> None:
  """Generates integer range constraint."""
  semantics: dict[str, typing.Any] = {
    "range_op": {
      "std_args": [{"name": "k", "type": "int", "min": 10, "max": 20}],
      "variants": {"torch": {"api": "foo"}, "jax": {"api": "bar"}},
    }
  }
  out_file: Path = tmp_path / "test_int_range.py"
  generator.generate(semantics, out_file)
  content: str = out_file.read_text()
  assert "random.randint(10, 20)" in content


def test_generate_float_range_constraint(generator: TestCaseGenerator, tmp_path: Path) -> None:
  """Generates float range constraint."""
  semantics: dict[str, typing.Any] = {
    "float_op": {
      "std_args": [{"name": "alpha", "type": "float", "min": 0.0, "max": 0.5}],
      "variants": {"torch": {"api": "foo"}, "jax": {"api": "bar"}},
    }
  }
  out_file: Path = tmp_path / "test_float_range.py"
  generator.generate(semantics, out_file)
  content: str = out_file.read_text()
  assert "random.uniform(0.0, 0.5)" in content


def test_generate_array_bounds_constraint(generator: TestCaseGenerator, tmp_path: Path) -> None:
  """Generates array bounds constraint."""
  semantics: dict[str, typing.Any] = {
    "sqrt": {
      "std_args": [{"name": "x", "type": "Array", "min": 0.001}],
      "variants": {"torch": {"api": "foo"}, "jax": {"api": "bar"}},
    }
  }
  out_file: Path = tmp_path / "test_array_bound.py"
  generator.generate(semantics, out_file)
  content: str = out_file.read_text()
  assert "np.abs(np.random.randn" in content
  assert "+ 0.001" in content


def test_generate_array_range_constraint(generator: TestCaseGenerator, tmp_path: Path) -> None:
  """Generates array range constraint."""
  semantics: dict[str, typing.Any] = {
    "limited": {
      "std_args": [{"name": "x", "type": "Array", "min": -1.0, "max": 1.0}],
      "variants": {"torch": {"api": "foo"}, "jax": {"api": "bar"}},
    }
  }
  out_file: Path = tmp_path / "test_array_range.py"
  generator.generate(semantics, out_file)
  content: str = out_file.read_text()
  assert "np.random.uniform(-1.0, 1.0" in content
