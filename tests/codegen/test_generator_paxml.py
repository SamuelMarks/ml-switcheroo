"""Test suite for the Generator Paxml module."""

from unittest.mock import MagicMock
from ml_switcheroo.generated_tests.generator import TestCaseGenerator
from ml_switcheroo.semantics.manager import SemanticsManager


def test_paxml_code_generation(tmp_path):
  """Verifies the behavior of Paxml code generation."""
  semantics = {"abs": {"std_args": ["x"], "variants": {"paxml": {"api": "jnp.abs"}, "numpy": {"api": "np.abs"}}}}
  mgr = MagicMock(spec=SemanticsManager)
  templates = {
    "paxml": {
      "import": "import praxis\nimport jax.numpy as jnp",
      "convert_input": "jnp.array({np_var})",
      "to_numpy": "np.array({res_var})",
    },
    "numpy": {"import": "import numpy as np", "convert_input": "{np_var}", "to_numpy": "{res_var}"},
  }
  mgr.get_test_template.side_effect = lambda fw: templates.get(fw)
  mgr.get_framework_config.return_value = {}
  mgr.test_templates = templates
  assert mgr.get_test_template("paxml") is not None, "PaxML template not loaded in Manager"
  gen = TestCaseGenerator(semantics_mgr=mgr)
  out_file = tmp_path / "test_pax_generated.py"
  gen.generate(semantics, out_file)
  assert out_file.exists()
  content = out_file.read_text(encoding="utf-8")
  runtime_file = out_file.parent / "runtime.py"
  assert runtime_file.exists()
  runtime_content = runtime_file.read_text(encoding="utf-8")
  assert 'find_spec("praxis")' in runtime_content
  assert 'find_spec("jax")' in runtime_content
  pax_block_start = content.find("Framework: paxml")
  assert pax_block_start != -1
  pax_block = content[pax_block_start:]
  assert "jnp.array(np_x)" in pax_block
  assert "results['paxml'] = np.array(res)" in pax_block
