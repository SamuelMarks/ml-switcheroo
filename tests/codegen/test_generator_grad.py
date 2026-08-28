"""Test suite for the Generator Grad module."""

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
    "torch": {"import": "import torch", "convert_input": "torch.tensor({np_var})", "to_numpy": "{res_var}.numpy()"},
    "jax": {"import": "import jax", "convert_input": "jnp.array({np_var})", "to_numpy": "{res_var}"},
  }
  mgr.get_test_template = MagicMock(side_effect=lambda fw: templates.get(fw))
  setattr(mgr, "test_templates", templates)
  mgr.get_framework_config = MagicMock(return_value={})
  return TestCaseGenerator(semantics_mgr=mgr)


def test_grad_injection_enabled(generator: TestCaseGenerator, tmp_path: Path) -> None:
  """Verifies the behavior of grad injection enabled."""
  semantics: dict[str, typing.Any] = {
    "sin": {"std_args": ["x"], "variants": {"torch": {"api": "torch.sin"}, "jax": {"api": "jnp.sin"}}}
  }
  out_file: Path = tmp_path / "test_sin.py"
  generator.generate(semantics, out_file)
  content: str = out_file.read_text()
  assert "jax.grad(lambda a0: jnp.sum(jnp.sin(a0)))(jnp.array(np_x))" in content
  assert "torch.func.grad(lambda a0: torch.sum(torch.sin(a0)))(torch.tensor(np_x))" in content
  assert "Gradient Verification" in content
  assert "np.testing.assert_allclose(g_ref, g_val" in content


def test_grad_injection_disabled_flag(generator: TestCaseGenerator, tmp_path: Path) -> None:
  """Verifies the behavior of grad injection disabled flag."""
  semantics: dict[str, typing.Any] = {
    "argmax": {
      "std_args": ["x"],
      "differentiable": False,
      "variants": {"torch": {"api": "torch.argmax"}, "jax": {"api": "jnp.argmax"}},
    }
  }
  out_file: Path = tmp_path / "test_nodiff.py"
  generator.generate(semantics, out_file)
  content: str = out_file.read_text()
  assert "jax.grad" not in content
  assert "Gradient Verification" not in content


def test_grad_injection_disabled_primitive(generator: TestCaseGenerator, tmp_path: Path) -> None:
  """Verifies the behavior of grad injection disabled primitive."""
  semantics: dict[str, typing.Any] = {
    "factorial": {
      "std_args": [{"name": "n", "type": "int"}],
      "variants": {"torch": {"api": "torch.math"}, "jax": {"api": "jax.math"}},
    }
  }
  out_file: Path = tmp_path / "test_int_input.py"
  generator.generate(semantics, out_file)
  content: str = out_file.read_text()
  assert "jax.grad" not in content


def test_grad_multi_arg(generator: TestCaseGenerator, tmp_path: Path) -> None:
  """Verifies the behavior of grad multi argument."""
  semantics: dict[str, typing.Any] = {
    "add": {"std_args": ["x", "y"], "variants": {"torch": {"api": "torch.add"}, "jax": {"api": "jnp.add"}}}
  }
  out_file: Path = tmp_path / "test_multi.py"
  generator.generate(semantics, out_file)
  content: str = out_file.read_text()
  assert "lambda a0, a1:" in content
  assert "jnp.add(a0, a1)" in content
