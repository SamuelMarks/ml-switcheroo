"""
Tests for Variable Backend Test Generation.

Verifies:
1.  Generator handles 'torch' and 'jax' correctly (standard case).
2.  Generator handles 'tensorflow' or 'mlx' if present in Semantics (dynamic case).
3.  Generator skips operations with only 1 variant.
4.  Generator respects manually existing tests.
"""

from ml_switcheroo.generated_tests.generator import TestGenerator


def test_generation_safety(tmp_path):
  """Verify manual overrides are respected."""
  # 1. Mock Semantics
  semantics = {"abs": {"variants": {"torch": {"api": "torch.abs"}, "jax": {"api": "jnp.abs"}}}}

  # 2. Setup existing file with a manual test
  out_file = tmp_path / "test_generated.py"
  out_file.write_text("""
def test_gen_abs():
    # Manual override
    assert True
""")

  # 3. Run Generator
  gen = TestGenerator()
  gen.generate(semantics, out_file)

  # 4. Verify Content
  content = out_file.read_text()
  # Should NOT contain the generated blocks
  assert "np.random.randn" not in content
  assert "results = {}" not in content


def test_generation_multi_backend(tmp_path):
  """
  Scenario: Semantics includes Torch, JAX, and TensorFlow.
  Expect: Generated code contains try/except blocks for all three.
  """
  # 1. Mock Semantics
  semantics = {
    "add": {
      "std_args": ["x", "y"],  # Binary
      "variants": {"torch": {"api": "torch.add"}, "jax": {"api": "jnp.add"}, "tensorflow": {"api": "tf.math.add"}},
    }
  }

  out_file = tmp_path / "test_multi.py"

  # 2. Run
  gen = TestGenerator()
  gen.generate(semantics, out_file)

  # 3. Verify Imports
  content = out_file.read_text()
  assert "import torch" in content
  assert "import tensorflow as tf" in content
  assert "import jax" in content

  # 4. Verify Execution Blocks
  assert "Framework: torch" in content
  assert "Framework: tensorflow" in content
  assert "tf.convert_to_tensor(np_x)" in content
  assert "res.numpy()" in content  # TF specific normalization


def test_excludes_single_variant(tmp_path):
  """
  Scenario: Operation only defined for Torch.
  Expect: No test generated (cannot compare).
  """
  semantics = {"unique_op": {"variants": {"torch": {"api": "torch.unique_thing"}}}}

  out_file = tmp_path / "test_empty.py"

  gen = TestGenerator()
  gen.generate(semantics, out_file)

  content = out_file.read_text() if out_file.exists() else ""
  # Header might be written if mode 'w', but no function body
  assert "def test_gen_unique_op" not in content


def test_generation_unary_vs_binary(tmp_path):
  """
  Verify argument count logic.
  """
  semantics = {
    "neg": {
      "std_args": ["x"],  # Unary
      "variants": {"torch": {"api": "torch.neg"}, "numpy": {"api": "np.negative"}},
    }
  }

  out_file = tmp_path / "test_unary.py"
  gen = TestGenerator()
  gen.generate(semantics, out_file)

  content = out_file.read_text()
  # Should not generate np_y
  assert "np_y" not in content
  # Call should be unary
  assert "torch.neg(torch.from_numpy(np_x))" in content
