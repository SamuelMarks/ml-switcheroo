"""Test suite for the Generator Types module."""

import typing
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ml_switcheroo.generated_tests.generator import TestCaseGenerator
from ml_switcheroo.generated_tests.inputs import generate_input_value_code
from ml_switcheroo.semantics.manager import SemanticsManager


@pytest.fixture
def gen(tmp_path: Path) -> TestCaseGenerator:
  """Docstring."""
  mgr: SemanticsManager = MagicMock(spec=SemanticsManager)

  def mock_get_template(fw: str) -> typing.Optional[dict[str, str]]:
    """Docstring."""
    if fw == "torch":
      return {"import": "import torch", "convert_input": "torch.tensor({np_var})", "to_numpy": "{res_var}.numpy()"}
    if fw == "jax":
      return {"import": "import jax", "convert_input": "jnp.array({np_var})", "to_numpy": "np.array({res_var})"}
    return None

  setattr(mgr, "get_test_template", MagicMock(side_effect=mock_get_template))
  return TestCaseGenerator(semantics_mgr=mgr)


def test_code_gen_str_int() -> None:
  """Verifies the behavior of code generation string integer."""
  code: str = generate_input_value_code("dim", "int")
  assert "random.randint" in code


def test_code_gen_str_bool() -> None:
  """Verifies the behavior of code generation string boolean."""
  code: str = generate_input_value_code("keepdims", "bool")
  assert "bool(random.getrandbits(1))" in code


def test_code_gen_str_float() -> None:
  """Verifies the behavior of code generation string float."""
  code: str = generate_input_value_code("alpha", "float")
  assert "random.uniform" in code


def test_code_gen_str_array() -> None:
  """Verifies the behavior of code generation string array."""
  code1: str = generate_input_value_code("x", "Array")
  assert "np.random.randn" in code1
  code2: str = generate_input_value_code("x", "Tensor")
  assert "np.random.randn" in code2


def test_code_gen_complex_list() -> None:
  """Verifies the behavior of code generation complex list."""
  code: str = generate_input_value_code("pads", "List[int]")
  assert "[1, 2]" in code


def test_code_gen_heuristic_fallback() -> None:
  """Verifies the behavior of code generation heuristic fallback."""
  code_axis: str = generate_input_value_code("axis", "Any")
  assert code_axis == "1"
  code_x: str = generate_input_value_code("x", "Any")
  assert "np.random.randn" in code_x


def test_generate_integration_typed_args(gen: TestCaseGenerator, tmp_path: Path) -> None:
  """Generates integration typed arguments."""
  semantics: dict[str, typing.Any] = {
    "randint_op": {
      "std_args": [("low", "int"), ("high", "int"), ("shape", "Tuple[int]")],
      "variants": {"torch": {"api": "torch.randint"}, "jax": {"api": "jax.random.randint"}},
    }
  }
  out_file: Path = tmp_path / "test_typed.py"
  gen.generate(semantics, out_file)
  content: str = out_file.read_text()
  assert "import random" in content
  assert "np_low = random.randint" in content
  assert "np_high = random.randint" in content
  assert "np_shape = (1, 2)" in content
  assert "np_low = np.random.randn" not in content


def test_return_type_verification_int(gen: TestCaseGenerator, tmp_path: Path) -> None:
  """Verifies the behavior of return type verification integer."""
  semantics: dict[str, typing.Any] = {
    "size_op": {"std_args": ["x"], "return_type": "int", "variants": {"torch": {"api": "foo"}, "jax": {"api": "bar"}}}
  }
  gen.generate(semantics, tmp_path / "test_int.py")
  content: str = (tmp_path / "test_int.py").read_text()
  assert "assert np.issubdtype(np.array(val).dtype, np.integer)" in content
  assert "or isinstance(val, int)" in content
  assert "Expected int" in content


def test_return_type_verification_bool(gen: TestCaseGenerator, tmp_path: Path) -> None:
  """Verifies the behavior of return type verification boolean."""
  semantics: dict[str, typing.Any] = {
    "is_nan": {"std_args": ["x"], "return_type": "bool", "variants": {"torch": {"api": "foo"}, "jax": {"api": "bar"}}}
  }
  gen.generate(semantics, tmp_path / "test_bool.py")
  content: str = (tmp_path / "test_bool.py").read_text()
  assert "assert np.issubdtype(np.array(val).dtype, bool)" in content
  assert "or isinstance(val, bool)" in content


def test_return_type_verification_tensor(gen: TestCaseGenerator, tmp_path: Path) -> None:
  """Verifies the behavior of return type verification tensor."""
  semantics: dict[str, typing.Any] = {
    "add": {"std_args": ["x"], "return_type": "Tensor", "variants": {"torch": {"api": "foo"}, "jax": {"api": "bar"}}}
  }
  gen.generate(semantics, tmp_path / "test_tensor.py")
  content: str = (tmp_path / "test_tensor.py").read_text()
  assert "assert isinstance(val, (np.ndarray, np.generic))" in content
  assert "Expected Array/Tensor" in content


def test_return_type_verification_other_float(gen: TestCaseGenerator, tmp_path: Path) -> None:
  """Verifies the behavior of return type verification for float or custom type."""
  semantics: dict[str, typing.Any] = {
    "norm": {"std_args": ["x"], "return_type": "float", "variants": {"torch": {"api": "foo"}, "jax": {"api": "bar"}}}
  }
  gen.generate(semantics, tmp_path / "test_float.py")
  content: str = (tmp_path / "test_float.py").read_text()
  assert "# Expected Array/Tensor" not in content
  assert "# Expected int" not in content
  assert "# Expected bool" not in content
