"""
Tests for JIT Compliance Generation (Feature 054).

Verifies:
1. JAX blocks contain `jax.jit`.
2. Static arguments (axis, keepdims) are correctly marked in `static_argnums`.
3. Other frameworks (Torch) do NOT contain `jax.jit`.
4. Lambda wrapping structure is syntactically valid for mulit-argument calls.
5. JIT flags are case-insensitive.
6. (New) Static keywords are loaded dynamically from Framework Config.
"""

from ml_switcheroo.generated_tests.generator import TestGenerator
from ml_switcheroo.semantics.manager import SemanticsManager
from unittest.mock import MagicMock


class MockTraitSemantics(SemanticsManager):
  """Ensures get_framework_config returns static args."""

  def get_framework_config(self, framework: str):
    if framework == "jax":
      return {"traits": {"jit_static_args": ["axis", "keepdims"]}}
    return {}

  # Also need to mock template retrieval if we use TestGenerator logic
  def get_test_template(self, fw):
    # Default behavior for tests
    if fw == "jax":
      return {"import": "import jax", "jit_wrap": "True"}
    if fw == "torch":
      return {"import": "import torch"}
    return None


def test_jax_block_has_jit(tmp_path):
  """
  Scenario: Verify generation for a unary JAX operation.
  Expect: 'jax.jit' appears in the code block.
  """
  semantics = {"abs": {"std_args": ["x"], "variants": {"jax": {"api": "jnp.abs"}, "torch": {"api": "torch.abs"}}}}

  # Inject mock manager
  mgr = MockTraitSemantics()

  out_file = tmp_path / "test_jit.py"
  gen = TestGenerator(semantics_mgr=mgr)
  gen.generate(semantics, out_file)

  content = out_file.read_text()

  # Check Imports
  assert "import jax" in content

  # Check JAX Block
  jax_section = content.split("Framework: jax")[1].split("except")[0]

  assert "jax.jit(fn" in jax_section
  assert "lambda a0: jnp.abs(a0)" in jax_section


def test_jit_static_argnums_detection(tmp_path):
  """
  Scenario: Operation has 'axis' argument (e.g. sum).
  Expect: jax.jit(fn, static_argnums=(1,))
  """
  semantics = {
    "sum": {
      "std_args": ["x", "axis"],
      "variants": {"jax": {"api": "jnp.sum"}, "torch": {"api": "torch.sum"}},
    }
  }

  mgr = MockTraitSemantics()
  out_file = tmp_path / "test_jit_static.py"
  gen = TestGenerator(semantics_mgr=mgr)
  gen.generate(semantics, out_file)

  content = out_file.read_text()
  jax_section = content.split("Framework: jax")[1].split("except")[0]

  # Verify static argnums injected because "axis" is in MockJAX static_args
  # x is index 0, axis is index 1.
  assert "static_argnums=(1,)" in jax_section


def test_jit_multiple_static_args(tmp_path):
  """
  Scenario: Op with multiple static args (x, axis, keepdims).
  Expect: static_argnums=(1, 2)
  """
  semantics = {
    "mean": {
      "std_args": ["x", "axis", "keepdims"],
      "variants": {"jax": {"api": "jnp.mean"}, "torch": {"api": "torch.mean"}},
    }
  }

  mgr = MockTraitSemantics()
  out_file = tmp_path / "test_jit_multi_static.py"
  gen = TestGenerator(semantics_mgr=mgr)
  gen.generate(semantics, out_file)

  content = out_file.read_text()
  jax_section = content.split("Framework: jax")[1]

  # Check for both indices
  assert "static_argnums=(1, 2)" in jax_section or "static_argnums=(1, 2,)" in jax_section


def test_torch_block_no_jit(tmp_path):
  """
  Scenario: Verify Torch block remains clean.
  """
  semantics = {"abs": {"std_args": ["x"], "variants": {"jax": {"api": "jnp.abs"}, "torch": {"api": "torch.abs"}}}}

  mgr = MockTraitSemantics()
  out_file = tmp_path / "test_jit_torch.py"
  gen = TestGenerator(semantics_mgr=mgr)
  gen.generate(semantics, out_file)

  content = out_file.read_text()

  torch_section = content.split("Framework: torch")[1].split("except")[0]

  assert "jit" not in torch_section
  assert "torch.abs(" in torch_section


def test_jit_flag_case_insensitivity(tmp_path):
  """
  Scenario: Semantics template sets "jit_wrap": "true" (lowercase).
  Expectation: Generator respects it as True.
  """
  # Mock a custom semantic manager returning lowercase true config
  mgr = MagicMock(spec=SemanticsManager)
  mgr.get_test_template.return_value = {
    "import": "import custom",
    "convert_input": "{np_var}",
    "to_numpy": "{res_var}",
    "jit_wrap": "true",  # Lowercase string
  }
  # Ensure no crash on config lookup
  mgr.get_framework_config.return_value = {}

  semantics = {
    "op": {
      "std_args": ["x"],
      "variants": {"custom": {"api": "custom.op"}, "torch": {"api": "t.op"}},
    }
  }

  out_file = tmp_path / "test_jit_case.py"
  gen = TestGenerator(semantics_mgr=mgr)
  gen.generate(semantics, out_file)

  content = out_file.read_text()
  custom_section = content.split("Framework: custom")[1]

  assert "jax.jit(fn" in custom_section


def test_jit_import_generation(tmp_path):
  """
  Scenario: 'jax' template is used.
  Expectation: File header should contain `import jax` even if only
  used implicitly by the JIT wrapper generator.
  """
  # We use MockTraitSemantics which defines JAX templates
  mgr = MockTraitSemantics()
  gen = TestGenerator(semantics_mgr=mgr)

  semantics = {
    "sum": {
      "variants": {"jax": {"api": "jnp.sum"}, "torch": {"api": "torch.sum"}},
      "std_args": ["x"],
    }
  }
  out_file = tmp_path / "test_imports.py"
  gen.generate(semantics, out_file)

  content = out_file.read_text()
  assert "import jax" in content


def test_jit_static_args_missing(tmp_path):
  """
  Scenario: Spec has 'dim' but config only lists 'axis'.
  Expect: No argnums added.
  """
  semantics = {"add": {"std_args": ["x", "dim"], "variants": {"jax": {"api": "jnp.add"}, "torch": {"api": "t.add"}}}}

  mgr = MockTraitSemantics()
  # Mock's get_framework_config returns ['axis', 'keepdims'] but NOT 'dim'

  out = tmp_path / "test_no_static.py"
  gen = TestGenerator(semantics_mgr=mgr)
  gen.generate(semantics, out)

  content = out.read_text()
  jax_sec = content.split("Framework: jax")[1]

  # Should wrap in jit because default template says so
  assert "jax.jit(fn)" in jax_sec  # No argnums arg
  assert "static_argnums" not in jax_sec
