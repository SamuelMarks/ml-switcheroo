"""
Tests for Template Persistence (Feature 2).

Verifies that:
1. Adapter registry provides baseline templates.
2. JSON configuration overrides registry templates.
"""

import json
from unittest.mock import patch
from ml_switcheroo.semantics.manager import SemanticsManager


def test_json_template_overrides_registry(tmp_path):
  """
  Scenario: User edits k_test_templates.json to change PyTorch syntax.
  Expectation: Manager returns the modified syntax, not the Adapter logic.
  """
  # 1. Create a custom template JSON
  template_file = tmp_path / "k_test_templates.json"
  custom_content = {
    "torch": {
      "import": "import torch as custom_torch",
      "convert_input": "custom_torch.tensor({np_var})",
      # We don't override to_numpy, so it should merge with Adapter defaults
    }
  }
  template_file.write_text(json.dumps(custom_content), encoding="utf-8")

  # 2. Patch resolution to point to our temp dir
  with patch("ml_switcheroo.semantics.manager.resolve_semantics_dir", return_value=tmp_path):
    mgr = SemanticsManager()

    # 3. Assert Override
    torch_tmpl = mgr.test_templates["torch"]
    # Should match our injected JSON
    assert torch_tmpl["import"] == "import torch as custom_torch"
    assert torch_tmpl["convert_input"] == "custom_torch.tensor({np_var})"

    # Should match Adapter default (merged)
    assert "detach().cpu()" in torch_tmpl["to_numpy"]


def test_registry_provides_defaults(tmp_path):
  """
  Scenario: No JSON provided.
  Expectation: Manager has templates from registered adapters.
  """
  # Point to empty dir
  with patch("ml_switcheroo.semantics.manager.resolve_semantics_dir", return_value=tmp_path):
    mgr = SemanticsManager()

    assert "torch" in mgr.test_templates
    assert "import torch" in mgr.test_templates["torch"]["import"]
