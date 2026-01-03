"""
Tests for the `define` CLI command handler.

Verifies:
1.  YAML parsing/validation errors are handled gracefully.
2.  Successful injection modifies Hub (Python) and Spoke (JSON) files properly.
3.  **Inference Integration**: Verifies 'infer' API triggers lookup.
"""

import sys
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from ml_switcheroo.cli.handlers.define import handle_define


@pytest.fixture
def mock_yaml_file(tmp_path):
  """Creates a valid ODL yaml file."""
  content = """
operation: "NewOp"
description: "A new operation"
std_args:
  - name: "x"
    type: "Tensor"
variants:
  mock_fw:
    api: "lib.op"
"""
  f = tmp_path / "new_op.yaml"
  f.write_text(content, encoding="utf-8")
  return f


@pytest.fixture
def infer_yaml_file(tmp_path):
  """Creates a YAML asking to infer API."""
  content = """
operation: "InferredOp"
description: "Auto Op"
std_args: []
variants:
  mock_fw:
    api: "infer"
"""
  f = tmp_path / "infer_op.yaml"
  f.write_text(content, encoding="utf-8")
  return f


@pytest.fixture
def mock_fs_env(tmp_path):
  """
  Creates a simulated filesystem.
  """
  # 1. Hub (Python file)
  semantics_dir = tmp_path / "semantics"
  semantics_dir.mkdir()
  hub_file = semantics_dir / "standards_internal.py"
  hub_file.write_text("INTERNAL_OPS = {}", encoding="utf-8")

  # 2. Spoke (JSON file)
  defs_dir = tmp_path / "frameworks" / "definitions"
  defs_dir.mkdir(parents=True)
  spoke_file = defs_dir / "mock_fw.json"
  spoke_file.write_text("{}", encoding="utf-8")

  return hub_file, spoke_file


def mock_env_patches(hub_file, spoke_file):
  """Applies patches for file locations."""
  # Patch 1: Hub location (inspect.getfile) -> standards_internal.py
  p1 = patch("inspect.getfile", return_value=str(hub_file))

  # Patch 2: Spoke location (get_definitions_path) -> mock_fw.json
  p2 = patch("ml_switcheroo.tools.injector_fw.core.get_definitions_path", return_value=spoke_file)

  # Patch 3: Adapter retrieval (must return something truthy so loop continues)
  mock_adapter = MagicMock()
  p3 = patch("ml_switcheroo.cli.handlers.define.get_adapter", return_value=mock_adapter)

  p1.start()
  p2.start()
  p3.start()
  return [p1, p2, p3]


def test_define_success_flow(mock_yaml_file, mock_fs_env):
  """
  Full integration: verify file updates on success.
  """
  hub_file, spoke_file = mock_fs_env
  patches = mock_env_patches(hub_file, spoke_file)

  try:
    # Handle missing YAML in env (if necessary)
    if "yaml" not in sys.modules:
      m_yaml = MagicMock()
      m_yaml.safe_load.return_value = {
        "operation": "NewOp",
        "description": "Desc",
        "std_args": [{"name": "x"}],
        "variants": {"mock_fw": {"api": "lib.op"}},
      }
      with patch("ml_switcheroo.cli.handlers.define.yaml", m_yaml):
        ret = handle_define(mock_yaml_file)
    else:
      ret = handle_define(mock_yaml_file)
  finally:
    for p in patches:
      p.stop()

  assert ret == 0

  # Verify Spoke Update (JSON)
  spoke_content = json.loads(spoke_file.read_text())
  assert "NewOp" in spoke_content
  assert spoke_content["NewOp"]["api"] == "lib.op"


def test_define_inference_flow(infer_yaml_file, mock_fs_env):
  """
  Integration: verify 'api: infer' triggers discovery and updates API.
  """
  hub_file, spoke_file = mock_fs_env
  patches = mock_env_patches(hub_file, spoke_file)

  # Mock Discovery
  with patch("ml_switcheroo.cli.handlers.define.SimulatedReflection") as MockReflect:
    ref_instance = MockReflect.return_value
    ref_instance.discover.return_value = "mock_fw.discovered_api"

    try:
      if "yaml" not in sys.modules:
        m_yaml = MagicMock()
        m_yaml.safe_load.return_value = {
          "operation": "InferredOp",
          "description": "Auto",
          "std_args": [],
          "variants": {"mock_fw": {"api": "infer"}},
        }
        with patch("ml_switcheroo.cli.handlers.define.yaml", m_yaml):
          handle_define(infer_yaml_file)
      else:
        handle_define(infer_yaml_file)
    finally:
      for p in patches:
        p.stop()

    # Verify Mock was called
    ref_instance.discover.assert_called_with("InferredOp")

  # Verify Spoke Update uses the DISCOVERED api in the JSON
  spoke_content = json.loads(spoke_file.read_text())
  assert "InferredOp" in spoke_content
  assert spoke_content["InferredOp"]["api"] == "mock_fw.discovered_api"
