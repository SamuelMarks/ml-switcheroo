"""
Tests for Template Persistence (Feature 2).

Verifies that:
1. JSON configuration overrides hardcoded defaults.
2. JSON configuration overrides Adapter code logic.
3. New frameworks defined in JSON are correctly loaded.
"""

import json
from unittest.mock import patch
from ml_switcheroo.semantics.manager import SemanticsManager


def test_json_template_overrides_defaults(tmp_path):
  """
  Scenario: User edits k_test_templates.json to change PyTorch syntax.
  Expectation: Manager returns the modified syntax, not the default or adapter logic.
  """
  # 1. Create a custom template JSON
  template_file = tmp_path / "k_test_templates.json"
  custom_content = {
    "torch": {
      "import": "import torch as custom_torch",
      "convert_input": "custom_torch.tensor({np_var})",
      "to_numpy": "{res_var}.numpy()",
    }
  }
  template_file.write_text(json.dumps(custom_content), encoding="utf-8")

  # 2. Patch resolution to point to our temp dir
  with patch("ml_switcheroo.semantics.manager.resolve_semantics_dir", return_value=tmp_path):
    mgr = SemanticsManager()

    # 3. Assert Override
    torch_tmpl = mgr.test_templates["torch"]
    # Should match our injected JSON, not the default "import torch"
    assert torch_tmpl["import"] == "import torch as custom_torch"
    assert torch_tmpl["convert_input"] == "custom_torch.tensor({np_var})"


def test_json_loads_new_framework(tmp_path):
  """
  Scenario: User adds a completely new backend (e.g. 'custom_backend') via JSON.
  Expectation: It appears in test_templates.
  """
  template_file = tmp_path / "k_test_templates.json"
  custom_content = {
    "custom_backend": {"import": "import custom", "convert_input": "custom.data({np_var})", "to_numpy": "{res_var}.data"}
  }
  template_file.write_text(json.dumps(custom_content), encoding="utf-8")

  with patch("ml_switcheroo.semantics.manager.resolve_semantics_dir", return_value=tmp_path):
    mgr = SemanticsManager()

    assert "custom_backend" in mgr.test_templates
    assert mgr.test_templates["custom_backend"]["import"] == "import custom"


def test_template_partial_override(tmp_path):
  """
  Scenario: JSON only overrides 'import', but leaves 'convert_input'.
  Logic: _merge_templates uses .update(), so keys should merge.
  """
  template_file = tmp_path / "k_test_templates.json"
  # Only overriding the import key for default torch template
  custom_content = {"torch": {"import": "import torch_custom"}}
  template_file.write_text(json.dumps(custom_content), encoding="utf-8")

  with patch("ml_switcheroo.semantics.manager.resolve_semantics_dir", return_value=tmp_path):
    mgr = SemanticsManager()

    torch_tmpl = mgr.test_templates["torch"]
    # Overridden
    assert torch_tmpl["import"] == "import torch_custom"
    # Preserved from defaults (because we only updated the dict)
    assert "convert_input" in torch_tmpl
    assert "torch.from_numpy" in torch_tmpl["convert_input"]
