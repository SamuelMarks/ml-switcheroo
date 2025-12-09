"""
Tests for PaxML Template Configuration.

Verifies that:
1. `k_test_templates.json` contains a valid entry for 'paxml'.
2. The template values generally match the `PaxmlAdapter` logic in code.
3. Placeholders ({np_var}, {res_var}) are present for synthesis.
"""

import json
from pathlib import Path
from ml_switcheroo.semantics.manager import resolve_semantics_dir, SemanticsManager


def test_paxml_template_exists_on_disk():
  """
  Verify the physical JSON file includes 'paxml'.
  """
  semantics_dir = resolve_semantics_dir()
  template_path = semantics_dir / "k_test_templates.json"

  assert template_path.exists(), "Templates JSON file missing."

  content = json.loads(template_path.read_text(encoding="utf-8"))

  assert "paxml" in content, "Missing 'paxml' key in k_test_templates.json"
  pax_conf = content["paxml"]

  # Verify structural requirements for TestGenerator
  assert "import" in pax_conf
  assert "convert_input" in pax_conf
  assert "to_numpy" in pax_conf


def test_paxml_template_content_logic():
  """
  Verify the content strings match expected JAX/Praxis semantics.
  """
  # Load via Manager to ensure standard loading flow works
  mgr = SemanticsManager()
  tmpl = mgr.get_test_template("paxml")

  assert tmpl is not None

  # 1. Imports: Must include praxis and jax (usually jnp alias)
  assert "import praxis" in tmpl["import"]
  assert "import jax.numpy as jnp" in tmpl["import"]

  # 2. Input Conversion: Should use jax numpy array creation
  assert "jnp.array" in tmpl["convert_input"]
  assert "{np_var}" in tmpl["convert_input"]

  # 3. Output Normalization: Should cast back via numpy
  assert "np.array" in tmpl["to_numpy"]
  assert "{res_var}" in tmpl["to_numpy"]


def test_paxml_adapter_sync_consistency():
  """
  Verify that the JSON definition aligns with the Python Adapter class code.
  This ensures no drift between the static Config and the runtime Adapter.
  """
  from ml_switcheroo.testing.adapters import PaxmlAdapter

  mgr = SemanticsManager()
  json_tmpl = mgr.get_test_template("paxml")

  # Adapter Logic
  code_import = PaxmlAdapter.get_import_stmts()
  code_convert = PaxmlAdapter.get_creation_syntax("{np_var}")
  code_output = PaxmlAdapter.get_numpy_conversion_syntax("{res_var}")

  # Assert basic equivalence (ignoring whitespace differences if any)
  assert json_tmpl["import"].strip() == code_import.strip()
  assert json_tmpl["convert_input"] == code_convert
  assert json_tmpl["to_numpy"] == code_output
