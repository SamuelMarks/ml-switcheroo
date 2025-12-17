"""
Tests for Custom Operation Discovery Script (scripts/discover_missing_ops.py).
"""

import types
import json
import pytest
import sys
from unittest.mock import patch, MagicMock
import importlib.util
from pathlib import Path

from ml_switcheroo.core.ghost import GhostRef, GhostParam

# Locate the script relative to this test file (tests/scripts/ -> scripts/)
SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "discover_missing_ops.py"


def load_script():
  """Loads the script file as a module."""
  if not SCRIPT_PATH.exists():
    pytest.skip(f"Script not found at {SCRIPT_PATH}")
  spec = importlib.util.spec_from_file_location("discover_missing_ops", SCRIPT_PATH)
  mod = importlib.util.module_from_spec(spec)
  sys.modules["discover_missing_ops"] = mod
  spec.loader.exec_module(mod)
  return mod


discover_ops = load_script()


@pytest.fixture
def mock_torch_env():
  """Sets up a mock torch environment."""
  mock_torch = types.ModuleType("torch")
  mock_nn = types.ModuleType("torch.nn")
  mock_functional = types.ModuleType("torch.nn.functional")

  mock_torch.nn = mock_nn
  mock_nn.functional = mock_functional

  # Create classes with metadata required by inspect
  class Linear:
    __module__ = "torch.nn"

    def __init__(self, in_features, out_features):
      pass

  class Conv2d:
    __module__ = "torch.nn"

    def __init__(self, in_channels, out_channels):
      pass

  def relu(x):
    pass

  # Populate the module dict
  mock_nn.Linear = Linear
  mock_nn.Conv2d = Conv2d
  mock_functional.relu = relu

  # Patch sys.modules so 'import torch.nn' works inside the script
  with patch.dict(sys.modules, {"torch": mock_torch, "torch.nn": mock_nn, "torch.nn.functional": mock_functional}):
    yield mock_nn


@pytest.fixture
def mock_flax_env():
  """Sets up a mock flax environment."""
  mock_flax = types.ModuleType("flax")
  mock_nnx = types.ModuleType("flax.nnx")

  mock_flax.nnx = mock_nnx

  class Linear:
    __module__ = "flax.nnx"

    def __init__(self, in_features, out_features):
      pass

  def relu(x):
    pass

  mock_nnx.Linear = Linear
  mock_nnx.relu = relu

  with patch.dict(sys.modules, {"flax": mock_flax, "flax.nnx": mock_nnx}):
    yield mock_nnx


def test_scan_torch_layers(mock_torch_env):
  """Verify torch scanner findings."""

  # Patch GhostInspector to return a valid Ref without doing real reflection
  with patch("ml_switcheroo.core.ghost.GhostInspector.inspect") as mock_inspect:

    def side_effect(obj, path):
      name = path.split(".")[-1]
      return GhostRef(name=name, api_path=path, kind="class", params=[])

    mock_inspect.side_effect = side_effect

    refs = discover_ops.scan_torch_layers()
    names = [r.name for r in refs]

    assert "Linear" in names
    assert "Conv2d" in names


def test_scan_torch_functional(mock_torch_env):
  """Verify torch functional scanner."""
  with patch("ml_switcheroo.core.ghost.GhostInspector.inspect") as mock_inspect:

    def side_effect(obj, path):
      name = path.split(".")[-1]
      return GhostRef(name=name, api_path=path, kind="function", params=[])

    mock_inspect.side_effect = side_effect

    refs = discover_ops.scan_torch_functional()
    names = [r.name for r in refs]

    assert "relu" in names


def test_e2e_discovery_execution(tmp_path, mock_torch_env, mock_flax_env):
  """
  Simulates full run writing to tmp_path by patching paths.
  """
  sem_dir = tmp_path / "semantics"
  snap_dir = tmp_path / "snapshots"
  sem_dir.mkdir()

  with patch("ml_switcheroo.semantics.paths.resolve_semantics_dir", return_value=sem_dir):
    with patch("ml_switcheroo.semantics.paths.resolve_snapshots_dir", return_value=snap_dir):
      with patch("ml_switcheroo.frameworks.available_frameworks", return_value=["torch", "flax_nnx"]):
        with patch("ml_switcheroo.frameworks.get_adapter", return_value=None):
          with patch("ml_switcheroo.core.ghost.GhostInspector.inspect") as mock_inspect:

            def side_effect(obj, path):
              name = path.split(".")[-1]
              kind = "function" if name[0].islower() else "class"
              return GhostRef(name=name, api_path=path, kind=kind, params=[GhostParam(name="x", kind="pos")])

            mock_inspect.side_effect = side_effect

            discover_ops.main()

  outfile = sem_dir / "k_discovered.json"
  assert outfile.exists()
  data = json.loads(outfile.read_text())

  # Both modules define Linear, so it should be present
  assert "Linear" in data
  # functional/relu is scanned, so it should be present
  assert "relu" in data
