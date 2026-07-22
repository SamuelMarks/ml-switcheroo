"""Test suite for the Generator Types module."""

import pytest
from unittest.mock import MagicMock
from ml_switcheroo.generated_tests.generator import TestCaseGenerator
from ml_switcheroo.generated_tests.inputs import generate_input_value_code
from ml_switcheroo.semantics.manager import SemanticsManager


@pytest.fixture
def gen(tmp_path):
  """Provides a mock generation for testing."""
  mgr = MagicMock(spec=SemanticsManager)

  def mock_get_template(fw):
    """Provides a mock get template for testing."""
    if fw == "torch":
      return {"import": "import torch", "convert_input": "torch.tensor({np_var})", "to_numpy": "{res_var}.numpy()"}
    if fw == "jax":
      return {"import": "import jax", "convert_input": "jnp.array({np_var})", "to_numpy": "np.array({res_var})"}
    return None

  mgr.get_test_template.side_effect = mock_get_template
  return TestCaseGenerator(semantics_mgr=mgr)


def test_code_gen_str_int():
  """Verifies the behavior of code generation string integer."""
  code = generate_input_value_code("dim", "int")
  assert "random.randint" in code


def test_code_gen_str_bool():
  """Verifies the behavior of code generation string boolean."""
  code = generate_input_value_code("keepdims", "bool")
  assert "bool(random.getrandbits(1))" in code


def test_code_gen_str_float():
  """Verifies the behavior of code generation string float."""
  code = generate_input_value_code("alpha", "float")
  assert "random.uniform" in code


def test_code_gen_str_array():
  """Verifies the behavior of code generation string array."""
  code1 = generate_input_value_code("x", "Array")
  assert "np.random.randn" in code1
  code2 = generate_input_value_code("x", "Tensor")
  assert "np.random.randn" in code2


def test_code_gen_complex_list():
  """Verifies the behavior of code generation complex list."""
  code = generate_input_value_code("pads", "List[int]")
  assert "[1, 2]" in code


def test_code_gen_heuristic_fallback():
  """Verifies the behavior of code generation heuristic fallback."""
  code_axis = generate_input_value_code("axis", "Any")
  assert code_axis == "1"
  code_x = generate_input_value_code("x", "Any")
  assert "np.random.randn" in code_x


def test_generate_integration_typed_args(gen, tmp_path):
  """Generates integration typed arguments."""
  semantics = {
    "randint_op": {
      "std_args": [("low", "int"), ("high", "int"), ("shape", "Tuple[int]")],
      "variants": {"torch": {"api": "torch.randint"}, "jax": {"api": "jax.random.randint"}},
    }
  }
  out_file = tmp_path / "test_typed.py"
  gen.generate(semantics, out_file)
  content = out_file.read_text()
  assert "import random" in content
  assert "np_low = random.randint" in content
  assert "np_high = random.randint" in content
  assert "np_shape = (1, 2)" in content
  assert "np_low = np.random.randn" not in content


def test_return_type_verification_int(gen, tmp_path):
  """Verifies the behavior of return type verification integer."""
  semantics = {
    "size_op": {"std_args": ["x"], "return_type": "int", "variants": {"torch": {"api": "foo"}, "jax": {"api": "bar"}}}
  }
  gen.generate(semantics, tmp_path / "test_int.py")
  content = (tmp_path / "test_int.py").read_text()
  assert "assert np.issubdtype(np.array(val).dtype, np.integer)" in content
  assert "or isinstance(val, int)" in content
  assert "Expected int" in content


def test_return_type_verification_bool(gen, tmp_path):
  """Verifies the behavior of return type verification boolean."""
  semantics = {
    "is_nan": {"std_args": ["x"], "return_type": "bool", "variants": {"torch": {"api": "foo"}, "jax": {"api": "bar"}}}
  }
  gen.generate(semantics, tmp_path / "test_bool.py")
  content = (tmp_path / "test_bool.py").read_text()
  assert "assert np.issubdtype(np.array(val).dtype, bool)" in content
  assert "or isinstance(val, bool)" in content


def test_return_type_verification_tensor(gen, tmp_path):
  """Verifies the behavior of return type verification tensor."""
  semantics = {
    "add": {"std_args": ["x"], "return_type": "Tensor", "variants": {"torch": {"api": "foo"}, "jax": {"api": "bar"}}}
  }
  gen.generate(semantics, tmp_path / "test_tensor.py")
  content = (tmp_path / "test_tensor.py").read_text()
  assert "assert isinstance(val, (np.ndarray, np.generic))" in content
  assert "Expected Array/Tensor" in content
