"""Module docstring."""

import numpy as np

from ml_switcheroo.frameworks.torch import TorchAdapter


def test_torch_no_torch_module(monkeypatch):
  """Docstring."""
  import sys

  monkeypatch.setitem(sys.modules, "torch", None)
  import importlib

  import ml_switcheroo.frameworks.torch as torch_mod

  try:
    importlib.reload(torch_mod)
  except Exception:
    pass


def test_torch_get_doc_url_init():
  """Docstring."""
  fw = TorchAdapter()
  assert "nn.init.html" in fw.get_doc_url("torch.nn.init.constant_")


def test_torch_convert_numpy_exception(monkeypatch):
  """Docstring."""
  fw = TorchAdapter()
  import torch

  # Let's mock built-in getattr or something to intercept `torch.from_numpy`?
  # actually, why don't we monkeypatch the REAL torch module in sys.modules

  def mock_from_numpy(data):
    """Docstring."""
    raise ValueError("mock error")

  monkeypatch.setattr(torch, "from_numpy", mock_from_numpy)

  res = fw.convert(np.array([1, 2]))
  assert isinstance(res, torch.Tensor)


def test_torch_convert_list_exception(monkeypatch):
  """Docstring."""
  fw = TorchAdapter()
  import torch

  def mock_tensor(data):
    """Docstring."""
    raise ValueError("mock error")

  monkeypatch.setattr(torch, "tensor", mock_tensor)
  res = fw.convert([1, 2, 3])
  # Should fall through and return the list
  assert res == [1, 2, 3]


def test_torch_collect_ghost_no_snapshot():
  """Docstring."""
  fw = TorchAdapter()
  fw._snapshot_data = None
  from ml_switcheroo_ir.schema.ghost import SemanticTier

  assert fw._collect_ghost(SemanticTier.LAYER) == []

  fw._snapshot_data = {"categories": {"layer": [{"api_path": "test", "name": "Test", "kind": "function"}]}}
  res = fw._collect_ghost(SemanticTier.LAYER)
  assert len(res) == 1
  assert res[0].api_path == "test"


def test_torch_collect_live_layer():
  """Docstring."""
  fw = TorchAdapter()
  from ml_switcheroo_ir.schema.ghost import SemanticTier

  def dummy_scan_layers():
    """Docstring."""
    return ["layer1"]

  fw._scan_layers = dummy_scan_layers
  res = fw._collect_live(SemanticTier.LAYER)
  assert res == ["layer1"]


def test_torch_import_fail_in_convert(monkeypatch):
  """Docstring."""
  import builtins

  fw = TorchAdapter()
  original_import = builtins.__import__

  def mock_import(name, *args, **kwargs):
    """Docstring."""
    if name == "torch":
      raise ImportError("mock error")
    return original_import(name, *args, **kwargs)

  monkeypatch.setattr(builtins, "__import__", mock_import)
  res = fw.convert([1, 2, 3])
  assert res == [1, 2, 3]


def test_torch_get_device_syntax():
  """Docstring."""
  fw = TorchAdapter()
  # Hit 221
  res = fw.get_device_syntax("xpu")
  assert "xpu" in res

  res = fw.get_device_syntax("cuda", "1")
  assert "1" in res


def test_torch_convert_numpy_fallback(monkeypatch):
  """Docstring."""
  fw = TorchAdapter()
  import sys

  class MockTorch:
    """Docstring."""

    def from_numpy(self, data):
      """Docstring."""
      raise ValueError()

    def tensor(self, data):
      """Docstring."""
      return "fallback"

  monkeypatch.setitem(sys.modules, "torch", MockTorch())
  res = fw.convert(np.array([1, 2]))
  assert res == "fallback"


def test_torch_convert_list_fallback(monkeypatch):
  """Docstring."""
  fw = TorchAdapter()
  import sys

  class MockTorch:
    """Docstring."""

    def tensor(self, data):
      """Docstring."""
      raise ValueError()

  monkeypatch.setitem(sys.modules, "torch", MockTorch())
  res = fw.convert([1, 2])
  assert res == [1, 2]


def test_torch_definitions_fallback(monkeypatch):
  """Docstring."""
  import ml_switcheroo.frameworks.torch as torch_mod

  monkeypatch.setattr(torch_mod, "load_definitions", lambda fw: {})
  fw = TorchAdapter()
  if callable(fw.definitions):
    defs = fw.definitions()
  else:
    defs = fw.definitions
  assert "ReLU" in defs
  assert "Linear" in defs
  assert "Conv2d" in defs
