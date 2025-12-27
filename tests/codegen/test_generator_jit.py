"""
Tests for JIT Compliance Generation (Feature 054).

Verifies:
1. JAX blocks contain `jax.jit` ONLY if explicitly templated.
2. Static arguments (axis, keepdims) are correctly marked in `static_argnums`.
3. Custom JIT templates (e.g. TinyGrad style) are rendered correctly.
4. Missing templates result in no JIT wrapping (Decoupling verification).
5. Lambda wrapping structure is syntactically valid for multi-argument calls.
"""

from ml_switcheroo.generated_tests.generator import TestGenerator
from ml_switcheroo.semantics.manager import SemanticsManager
from unittest.mock import MagicMock


class MockTraitSemantics(SemanticsManager):
  """Ensures get_framework_config returns static args."""

  def __init__(self, templates=None):
    self.custom_templates = templates or {}

  def get_framework_config(self, framework: str):
    if framework == "jax":
      return {"traits": {"jit_static_args": ["axis", "keepdims"]}}
    if framework == "tinygrad":
      # TinyGrad generally handles static args differently but let's mock it having some
      return {"traits": {"jit_static_args": ["axis"]}}
    return {}

  def get_test_template(self, fw):
    if fw in self.custom_templates:
      return self.custom_templates[fw]

    # Default behavior for tests if not overridden
    if fw == "jax":
      # Mock the standard JAX template which includes JIT
      return {
        "import": "import jax",
        "jit_template": "jax.jit({fn}, static_argnums={static_argnums})",
        "convert_input": "{np_var}",
        "to_numpy": "{res_var}",
      }
    if fw == "torch":
      return {"import": "import torch", "convert_input": "torch.tensor({np_var})", "to_numpy": "{res_var}.numpy()"}
    return None


def test_missing_template_skips_jit(tmp_path):
  """
  Scenario: Template has NO 'jit_template'.
  Expect: No JIT wrapping, just direct call. (Verifies Decoupling).
  """
  semantics = {"abs": {"std_args": ["x"], "variants": {"jax": {"api": "jnp.abs"}, "torch": {"api": "torch.abs"}}}}

  # Inject mock manager with explicit NO template config for JAX
  no_jit_tmpl = {
    "jax": {"import": "import jax", "convert_input": "{np_var}", "to_numpy": "{res_var}"},
    "torch": {"import": "import torch"},
  }

  mgr = MockTraitSemantics(templates=no_jit_tmpl)

  out_file = tmp_path / "test_jit_skipped.py"
  gen = TestGenerator(semantics_mgr=mgr)
  gen.generate(semantics, out_file)

  content = out_file.read_text()

  # Check JAX Block
  jax_section = content.split("Framework: jax")[1].split("except")[0]

  # Should use direct call, NOT jax.jit
  assert "jax.jit(" not in jax_section
  assert "jnp.abs(" in jax_section


def test_standard_jit_template(tmp_path):
  """
  Scenario: Standard JAX template with {fn} and {static_argnums}.
  Expect: jax.jit(fn, static_argnums=None) for non-static calls.
  """
  semantics = {"abs": {"std_args": ["x"], "variants": {"jax": {"api": "jnp.abs"}, "torch": {"api": "torch.abs"}}}}

  mgr = MockTraitSemantics()  # Uses default mock which has jit_template
  out_file = tmp_path / "test_jit_std.py"
  gen = TestGenerator(semantics_mgr=mgr)
  gen.generate(semantics, out_file)

  content = out_file.read_text()
  jax_section = content.split("Framework: jax")[1].split("except")[0]

  assert "jax.jit" in jax_section
  assert "static_argnums=None" in jax_section


def test_custom_jit_template(tmp_path):
  """
  Scenario: Template provides "jit_template": "TinyJit.trace({fn})"
  Expect: Generator uses the custom string.
  """
  semantics = {"add": {"std_args": ["x", "y"], "variants": {"tinygrad": {"api": "add"}, "torch": {"api": "add"}}}}

  custom_tmpl = {
    "tinygrad": {
      "import": "import tinygrad",
      "jit_template": "TinyJit.trace({fn})",
      "convert_input": "{np_var}",
      "to_numpy": "{res_var}.numpy()",
    },
    "torch": {"import": "import torch"},
  }

  mgr = MockTraitSemantics(templates=custom_tmpl)

  out_file = tmp_path / "test_jit_custom.py"
  gen = TestGenerator(semantics_mgr=mgr)
  gen.generate(semantics, out_file)

  content = out_file.read_text()
  tiny_section = content.split("Framework: tinygrad")[1].split("except")[0]

  assert "TinyJit.trace(fn)" in tiny_section
  assert "jax.jit" not in tiny_section


def test_jit_static_argnums_detection(tmp_path):
  """
  Scenario: Operation has 'axis' argument (e.g. sum), matched against template.
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
  assert "static_argnums=(1,)" in jax_section


def test_custom_template_static_args_interpolation(tmp_path):
  """
  Scenario: Template uses {static_argnums} placeholder.
  Expect: "custom_jit(fn, static=(1,))"
  """
  semantics = {
    "sum": {
      "std_args": ["x", "axis"],
      "variants": {"tinygrad": {"api": "sum"}, "torch": {"api": "sum"}},
    }
  }

  custom_tmpl = {
    "tinygrad": {
      "import": "import tinygrad",
      # Define specific slot for static args
      "jit_template": "custom_jit({fn}, static={static_argnums})",
    },
    "torch": {"import": "import torch"},
  }

  mgr = MockTraitSemantics(templates=custom_tmpl)
  out_file = tmp_path / "test_jit_custom_static.py"

  gen = TestGenerator(semantics_mgr=mgr)
  gen.generate(semantics, out_file)

  content = out_file.read_text()
  tiny_section = content.split("Framework: tinygrad")[1].split("except")[0]

  # Check interpolation
  assert "custom_jit(fn, static=(1,))" in tiny_section


def test_custom_template_static_args_missing(tmp_path):
  """
  Scenario: Template expects {static_argnums} but op has none.
  Expect: Placeholder replaced with 'None'.
  """
  semantics = {
    "abs": {
      "std_args": ["x"],  # No static args like axis
      "variants": {"tinygrad": {"api": "abs"}, "torch": {"api": "abs"}},
    }
  }

  custom_tmpl = {
    "tinygrad": {
      "import": "import tinygrad",
      "jit_template": "custom_jit({fn}, static={static_argnums})",
    },
    "torch": {"import": "import torch"},
  }

  mgr = MockTraitSemantics(templates=custom_tmpl)
  out_file = tmp_path / "test_jit_custom_none.py"

  gen = TestGenerator(semantics_mgr=mgr)
  gen.generate(semantics, out_file)

  content = out_file.read_text()
  tiny_section = content.split("Framework: tinygrad")[1].split("except")[0]

  # Check default replacement
  assert "custom_jit(fn, static=None)" in tiny_section


def test_torch_block_no_jit(tmp_path):
  """
  Scenario: Verify Torch block remains clean (no jit templates defined).
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


def test_jit_import_generation(tmp_path):
  """
  Scenario: 'jax' template is used.
  Expectation: File header should contain `import jax` defined in the template config.
  """
  # Mock manager behavior implicitly uses default JAX template
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
