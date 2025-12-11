"""
Tests for Registry Template Syncing.

Verifies that:
1. `TemplateGenerator` extracts strings from `FrameworkAdapter` classes.
2. Custom registered adapters are auto-discovered.
3. `SemanticsManager` effectively updates itself with these templates.
"""

import pytest
from ml_switcheroo.testing.registry_sync import TemplateGenerator
from ml_switcheroo.testing.adapters import register_adapter, _ADAPTER_REGISTRY
from ml_switcheroo.semantics.manager import SemanticsManager


class MockAdapterWithTemplates:
  """A valid adapter implementing the TemplateProvider protocol."""

  def convert(self, data):
    return data

  @classmethod
  def get_import_stmts(cls):
    return "import my_custom_lib"

  @classmethod
  def get_creation_syntax(cls, var_name):
    return f"my_custom_lib.tensor({var_name})"

  @classmethod
  def get_numpy_conversion_syntax(cls, var_name):
    return f"{var_name}.to_numpy()"


@pytest.fixture
def clean_registry():
  """Backup and restore registry."""
  original = _ADAPTER_REGISTRY.copy()
  yield
  _ADAPTER_REGISTRY.clear()
  _ADAPTER_REGISTRY.update(original)


def test_generate_from_default_registry(clean_registry):
  """
  Verify that standard adapters (Torch, JAX) produce templates.
  """
  templates = TemplateGenerator.generate_templates()

  assert "torch" in templates
  assert templates["torch"]["import"] == "import torch"
  assert "torch.from_numpy" in templates["torch"]["convert_input"]

  assert "jax" in templates
  assert "import jax" in templates["jax"]["import"]


def test_custom_adapter_sync(clean_registry):
  """
  Verify that registering a new class updates the generator output.
  """
  register_adapter("custom_fw", MockAdapterWithTemplates)

  templates = TemplateGenerator.generate_templates()

  assert "custom_fw" in templates
  tmpl = templates["custom_fw"]
  assert tmpl["import"] == "import my_custom_lib"
  assert tmpl["convert_input"] == "my_custom_lib.tensor({np_var})"
  assert tmpl["to_numpy"] == "{res_var}.to_numpy()"


def test_ignore_partial_adapter(clean_registry):
  """
  Verify that adapters missing methods are skipped.
  """

  class PartialAdapter:
    # Missing get_import_stmts etc.
    def convert(self, data):
      return data

  register_adapter("broken", PartialAdapter)

  templates = TemplateGenerator.generate_templates()
  assert "broken" not in templates


def test_semantics_manager_auto_load(clean_registry):
  """
  Verify SemanticsManager calls the sync on init.
  """
  register_adapter("auto_sync_fw", MockAdapterWithTemplates)

  mgr = SemanticsManager()
  mgr._reverse_index = {}

  # Should be present in manager.test_templates
  assert "auto_sync_fw" in mgr.test_templates
  assert mgr.test_templates["auto_sync_fw"]["import"] == "import my_custom_lib"
