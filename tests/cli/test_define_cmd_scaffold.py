"""
Tests for CLI 'define' command processing Plugin Scaffolding.
Integration test verifying YAML -> DSL -> Generator.
"""

import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import ml_switcheroo.plugins

from ml_switcheroo.cli.handlers.define import handle_define


@pytest.fixture
def scaffold_yaml(tmp_path):
  """Creates a YAML asking for a plugin."""
  content = """ 
operation: "ComplexOp" 
description: "Op needing plugin" 
std_args: [] 
variants: 
  torch: 
    api: "torch.complex" 
scaffold_plugins: 
  - name: "complex_plugin" 
    type: "call_transform" 
    doc: "Handles complex logic" 
"""
  f = tmp_path / "complex.yaml"
  f.write_text(content)
  return f


@pytest.fixture
def mock_env(tmp_path):
  """
  Mocks filesystem for Hub and Spoke to allow the command to pass
  early checks and reach the plugin stage.
  """
  src = tmp_path / "src"
  src.mkdir()

  # Hub
  (src / "standards.py").write_text("INTERNAL_OPS = {}", "utf-8")

  # Spoke
  (src / "torch.py").write_text(
    "@register_framework('torch')\nclass A:\n @property\ndef definitions(self):\n return {}", "utf-8"
  )

  return src


def test_define_scaffolds_plugin(scaffold_yaml, mock_env, tmp_path):
  """
  Verify `handle_define` calls PluginGenerator and creates file.
  We patch `inspect` to point hub/spoke logic to our mock_env,
  and we point the plugin dir to a temp dir.
  """
  hub_file = mock_env / "standards.py"
  spoke_file = mock_env / "torch.py"
  plugins_out = tmp_path / "plugins_out"
  plugins_out.mkdir()

  # Mock inspect to pass hub/spoke checks
  with patch("inspect.getfile") as mock_file:

    def getfile(obj):
      name = getattr(obj, "__name__", "")
      if "standards_internal" in name:
        return str(hub_file)
      if "TorchAdapter" in name:  # returned by get_adapter mock below
        return str(spoke_file)
      # Default fallthrough for other modules like 'os'
      return __file__

    mock_file.side_effect = getfile

    # Mock adapter retrieval
    mock_adp = MagicMock()
    type(mock_adp).__name__ = "TorchAdapter"

    with patch("ml_switcheroo.cli.handlers.define.get_adapter", return_value=mock_adp):
      # Mock the location of ml_switcheroo.plugins to our temp dir
      # We patch the attribute on the imported module object used in define.py
      with patch.object(ml_switcheroo.plugins, "__file__", str(plugins_out / "__init__.py")):
        # Mock yaml availability
        m_yaml = MagicMock()
        # Simple parser for the fixture above
        data = {
          "operation": "ComplexOp",
          "description": "Desc",
          "std_args": [],
          "variants": {"torch": {"api": "torch.complex"}},
          "scaffold_plugins": [{"name": "complex_plugin", "type": "call_transform", "doc": "Handles complex logic"}],
        }
        m_yaml.safe_load_all.return_value = [data]
        m_yaml.safe_load.return_value = data

        # Force YAML injection since we can't rely on real PyYAML in all test envs
        with patch("ml_switcheroo.cli.handlers.define.yaml", m_yaml):
          ret = handle_define(scaffold_yaml)

          assert ret == 0

          # Verify Plugin Creation
          plugin_file = plugins_out / "complex_plugin.py"
          assert plugin_file.exists()
          content = plugin_file.read_text("utf-8")
          assert '@register_hook("complex_plugin")' in content
