"""Test suite for the Generator Jit module."""

from ml_switcheroo.generated_tests.generator import TestCaseGenerator
from ml_switcheroo.semantics.manager import SemanticsManager


class MockTraitSemantics(SemanticsManager):
  """Mock Trait Semantics class for testing purposes."""

  def __init__(self, templates=None):
    """Initializes the MockTraitSemantics instance."""
    self.custom_templates = templates or {}
    self.test_templates = self.custom_templates

  def get_framework_config(self, framework: str):
    """Mock implementation of get framework configuration."""
    if framework == "jax":
      return {"traits": {"jit_static_args": ["axis", "keepdims"]}}
    if framework == "tinygrad":
      return {"traits": {"jit_static_args": ["axis"]}}
    return {}

  def get_test_template(self, fw):
    """Mock implementation of get test template."""
    if fw in self.custom_templates:
      return self.custom_templates[fw]
    if fw == "jax":
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
  """Verifies the behavior of missing template skips jit."""
  semantics = {"abs": {"std_args": ["x"], "variants": {"jax": {"api": "jnp.abs"}, "torch": {"api": "torch.abs"}}}}
  no_jit_tmpl = {
    "jax": {"import": "import jax", "convert_input": "{np_var}", "to_numpy": "{res_var}"},
    "torch": {"import": "import torch"},
  }
  mgr = MockTraitSemantics(templates=no_jit_tmpl)
  mgr.test_templates = no_jit_tmpl
  out_file = tmp_path / "test_jit_skipped.py"
  gen = TestCaseGenerator(semantics_mgr=mgr)
  gen.generate(semantics, out_file)
  content = out_file.read_text()
  jax_section = content.split("Framework: jax")[1].split("except")[0]
  assert "jax.jit(" not in jax_section
  assert "jnp.abs(" in jax_section


def test_standard_jit_template(tmp_path):
  """Verifies the behavior of standard jit template."""
  semantics = {"abs": {"std_args": ["x"], "variants": {"jax": {"api": "jnp.abs"}, "torch": {"api": "torch.abs"}}}}
  mgr = MockTraitSemantics()
  mgr.test_templates = {"jax": {"import": "import jax"}, "torch": {"import": "import torch"}}
  out_file = tmp_path / "test_jit_std.py"
  gen = TestCaseGenerator(semantics_mgr=mgr)
  gen.generate(semantics, out_file)
  content = out_file.read_text()
  jax_section = content.split("Framework: jax")[1].split("except")[0]
  assert "jax.jit" in jax_section
  assert "static_argnums=None" in jax_section


def test_custom_jit_template(tmp_path):
  """Verifies the behavior of custom jit template."""
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
  mgr.test_templates = custom_tmpl
  out_file = tmp_path / "test_jit_custom.py"
  gen = TestCaseGenerator(semantics_mgr=mgr)
  gen.generate(semantics, out_file)
  content = out_file.read_text()
  tiny_section = content.split("Framework: tinygrad")[1].split("except")[0]
  assert "TinyJit.trace(fn)" in tiny_section
  assert "jax.jit" not in tiny_section


def test_jit_static_argnums_detection(tmp_path):
  """Verifies the behavior of jit static argnums detection."""
  semantics = {"sum": {"std_args": ["x", "axis"], "variants": {"jax": {"api": "jnp.sum"}, "torch": {"api": "torch.sum"}}}}
  mgr = MockTraitSemantics()
  mgr.test_templates = {"jax": {"import": "import jax"}, "torch": {"import": "import torch"}}
  out_file = tmp_path / "test_jit_static.py"
  gen = TestCaseGenerator(semantics_mgr=mgr)
  gen.generate(semantics, out_file)
  content = out_file.read_text()
  jax_section = content.split("Framework: jax")[1].split("except")[0]
  assert "static_argnums=(1,)" in jax_section


def test_custom_template_static_args_interpolation(tmp_path):
  """Verifies the behavior of custom template static arguments interpolation."""
  semantics = {"sum": {"std_args": ["x", "axis"], "variants": {"tinygrad": {"api": "sum"}, "torch": {"api": "sum"}}}}
  custom_tmpl = {
    "tinygrad": {"import": "import tinygrad", "jit_template": "custom_jit({fn}, static={static_argnums})"},
    "torch": {"import": "import torch"},
  }
  mgr = MockTraitSemantics(templates=custom_tmpl)
  mgr.test_templates = custom_tmpl
  out_file = tmp_path / "test_jit_custom_static.py"
  gen = TestCaseGenerator(semantics_mgr=mgr)
  gen.generate(semantics, out_file)
  content = out_file.read_text()
  tiny_section = content.split("Framework: tinygrad")[1].split("except")[0]
  assert "custom_jit(fn, static=(1,))" in tiny_section


def test_custom_template_static_args_missing(tmp_path):
  """Verifies the behavior of custom template static arguments missing."""
  semantics = {"abs": {"std_args": ["x"], "variants": {"tinygrad": {"api": "abs"}, "torch": {"api": "abs"}}}}
  custom_tmpl = {
    "tinygrad": {"import": "import tinygrad", "jit_template": "custom_jit({fn}, static={static_argnums})"},
    "torch": {"import": "import torch"},
  }
  mgr = MockTraitSemantics(templates=custom_tmpl)
  mgr.test_templates = custom_tmpl
  out_file = tmp_path / "test_jit_custom_none.py"
  gen = TestCaseGenerator(semantics_mgr=mgr)
  gen.generate(semantics, out_file)
  content = out_file.read_text()
  tiny_section = content.split("Framework: tinygrad")[1].split("except")[0]
  assert "custom_jit(fn, static=None)" in tiny_section


def test_torch_block_no_jit(tmp_path):
  """Verifies the behavior of PyTorch block no jit."""
  semantics = {"abs": {"std_args": ["x"], "variants": {"jax": {"api": "jnp.abs"}, "torch": {"api": "torch.abs"}}}}
  mgr = MockTraitSemantics()
  mgr.test_templates = {"jax": {"import": "import jax"}, "torch": {"import": "import torch"}}
  out_file = tmp_path / "test_jit_torch.py"
  gen = TestCaseGenerator(semantics_mgr=mgr)
  gen.generate(semantics, out_file)
  content = out_file.read_text()
  torch_section = content.split("Framework: torch")[1].split("except")[0]
  assert "jit" not in torch_section
  assert "torch.abs(" in torch_section


def test_runtime_generation_includes_jit_modules(tmp_path):
  """Verifies the behavior of runtime generation includes jit modules."""
  semantics = {"sum": {"variants": {"jax": {"api": "jnp.sum"}, "torch": {"api": "torch.sum"}}, "std_args": ["x"]}}
  mgr = MockTraitSemantics()
  mgr.test_templates = {"jax": {"import": "import jax"}, "torch": {"import": "import torch"}}
  gen = TestCaseGenerator(semantics_mgr=mgr)
  out_file = tmp_path / "subdir" / "test.py"
  gen.generate(semantics, out_file)
  runtime = out_file.parent / "runtime.py"
  assert runtime.exists()
  assert 'find_spec("jax")' in runtime.read_text()
