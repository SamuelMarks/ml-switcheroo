"""
test_generator_paxml.py

Verifies that the TestGenerator correctly handles the 'paxml' framework configuration.
Ensures that templates derived from `k_test_templates.json` (or Registry Sync)
are correctly applied to generate valid executable test code including:
1. Specific imports (praxis, jax).
2. Input conversion syntax (jnp.array).
3. Output normalization syntax (np.array).
"""

import pytest
from unittest.mock import MagicMock
from ml_switcheroo.generated_tests.generator import TestGenerator
from ml_switcheroo.semantics.manager import SemanticsManager


def test_paxml_code_generation(tmp_path):
  """
  Scenario:
      Semantics definition contains a 'paxml' variant.
  Expected Output:
      - A test file containing `import praxis`.
      - A try/except block for `Framework: paxml`.
      - Conversion of numpy inputs using `jnp.array(...)`.
  """
  # 1. Setup Mock Semantics Data
  # The generator needs at least 2 variants to create a comparison test.
  semantics = {"abs": {"std_args": ["x"], "variants": {"paxml": {"api": "jnp.abs"}, "numpy": {"api": "np.abs"}}}}

  # 2. Logic Setup
  # Mock the Manager to provide templates without relying on disk state/bootstrap
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

  # Sanity check: Ensure template is actually loaded before generating
  # If this fails, the issue is in Manager loading, not Generator logic.
  assert mgr.get_test_template("paxml") is not None, "PaxML template not loaded in Manager"

  gen = TestGenerator(semantics_mgr=mgr)
  out_file = tmp_path / "test_pax_generated.py"

  # 3. Generate Code
  gen.generate(semantics, out_file)

  # 4. Verification
  assert out_file.exists()
  content = out_file.read_text(encoding="utf-8")

  # Check Header Imports
  assert "import praxis" in content
  assert "import jax.numpy as jnp" in content

  # Check specific test function body
  pax_block_start = content.find("Framework: paxml")
  assert pax_block_start != -1

  # Extract the paxml block for stricter local checking
  pax_block = content[pax_block_start:]

  # Input Conversion: jnp.array(np_x)
  assert "jnp.array(np_x)" in pax_block

  # Output Normalization: np.array(res)
  assert "results['paxml'] = np.array(res)" in pax_block
