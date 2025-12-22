"""
Tests for the `define` CLI command handler.

Verifies:
1.  YAML parsing/validation errors are handled gracefully.
2.  Missing files (hub source or adapter source) are handled.
3.  Successful injection modifies files properly.
4.  **Inference Integration**: Verifies 'infer' API triggers lookup.
"""

import sys
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
def mock_source_env(tmp_path):
  """
  Creates a simulated source tree with a standards file and adapter file.
  """
  semantics_dir = tmp_path / "src" / "semantics"
  semantics_dir.mkdir(parents=True)

  standards_file = semantics_dir / "standards_internal.py"
  standards_file.write_text("INTERNAL_OPS = {}", encoding="utf-8")

  frameworks_dir = tmp_path / "src" / "frameworks"
  frameworks_dir.mkdir(parents=True)

  adapter_file = frameworks_dir / "mock_fw.py"
  adapter_file.write_text(
    """
from ml_switcheroo.frameworks.base import register_framework, StandardMap

@register_framework("mock_fw")
class MockAdapter:
    @property
    def definitions(self):
        return {}
""",
    encoding="utf-8",
  )

  return standards_file, adapter_file


def mock_inspect_env(hub_file, spoke_file):
  """Applies patches for file inspection logic."""
  patcher = patch("inspect.getfile")
  mock_getfile = patcher.start()

  def side_effect(obj):
    if getattr(obj, "__name__", "") == "ml_switcheroo.semantics.standards_internal":
      return str(hub_file)
    if getattr(obj, "__name__", "") == "MockAdapterCls":
      return str(spoke_file)
    return str(hub_file)

  mock_getfile.side_effect = side_effect
  return patcher


def test_define_success_flow(mock_yaml_file, mock_source_env):
  """
  Full integration: verify file updates on success.
  """
  hub_file, spoke_file = mock_source_env

  # 1. Setup Inspect Patch
  insp_patch = mock_inspect_env(hub_file, spoke_file)

  # 2. Mock Adapter
  mock_adapter_instance = MagicMock()
  type(mock_adapter_instance).__name__ = "MockAdapterCls"

  with patch(
    "ml_switcheroo.cli.handlers.define.get_adapter", lambda k: mock_adapter_instance if k == "mock_fw" else None
  ):
    # 3. Handle missing YAML in env (if necessary)
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

  insp_patch.stop()
  assert ret == 0

  spoke_code = spoke_file.read_text()
  assert '"NewOp": StandardMap(api="lib.op")' in spoke_code


def test_define_inference_flow(infer_yaml_file, mock_source_env):
  """
  Integration: verify 'api: infer' triggers discovery and updates API.
  """
  hub_file, spoke_file = mock_source_env
  insp_patch = mock_inspect_env(hub_file, spoke_file)

  mock_adapter_instance = MagicMock()
  type(mock_adapter_instance).__name__ = "MockAdapterCls"

  # Mock Discovery so we don't rely on real frameworks
  with patch("ml_switcheroo.cli.handlers.define.SimulatedReflection") as MockReflect:
    ref_instance = MockReflect.return_value
    ref_instance.discover.return_value = "mock_fw.discovered_api"

    with patch(
      "ml_switcheroo.cli.handlers.define.get_adapter", lambda k: mock_adapter_instance if k == "mock_fw" else None
    ):
      # Ensure YAML loading works
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

    # Verify Mock was called
    ref_instance.discover.assert_called_with("InferredOp")

  insp_patch.stop()

  # Verify Spoke Update uses the DISCOVERED api, not 'infer'
  spoke_code = spoke_file.read_text()
  assert 'api="mock_fw.discovered_api"' in spoke_code
  assert 'api="infer"' not in spoke_code


def test_define_inference_failure_skips_injection(infer_yaml_file, mock_source_env):
  """
  Integration: verify 'api: infer' failing means no spoke update.
  """
  hub_file, spoke_file = mock_source_env
  insp_patch = mock_inspect_env(hub_file, spoke_file)
  mock_adapter_instance = MagicMock()
  type(mock_adapter_instance).__name__ = "MockAdapterCls"

  with patch("ml_switcheroo.cli.handlers.define.SimulatedReflection") as MockReflect:
    # Simulate Failure
    MockReflect.return_value.discover.return_value = None

    with patch("ml_switcheroo.cli.handlers.define.get_adapter", lambda k: mock_adapter_instance):
      if "yaml" not in sys.modules:
        m_yaml = MagicMock()
        m_yaml.safe_load.return_value = {
          "operation": "Op",
          "description": "",
          "std_args": [],
          "variants": {"mock_fw": {"api": "infer"}},
        }
        with patch("ml_switcheroo.cli.handlers.define.yaml", m_yaml):
          handle_define(infer_yaml_file)
      else:
        handle_define(infer_yaml_file)

  insp_patch.stop()

  # Verify Spoke Update SKIPPED
  spoke_code = spoke_file.read_text()
  assert "InferredOp" not in spoke_code
