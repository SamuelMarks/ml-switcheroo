import pytest
import sys
from unittest.mock import MagicMock, patch
from ml_switcheroo.frameworks.mlx import MLXAdapter
from ml_switcheroo.frameworks.base import StandardCategory
import ml_switcheroo.frameworks.mlx as mlx_mod


def test_mlx_convert_no_import(monkeypatch):
  old_mlx = sys.modules.get("mlx.core")
  sys.modules["mlx.core"] = None
  adapter = MLXAdapter()

  # 302-309
  res = adapter.convert([1, 2, 3])
  # if mlx isn't installed it skips or raises

  # 257-284, 287-290: discover without mx
  res_layer = adapter.collect_api(StandardCategory.LAYER)
  assert res_layer == []

  if old_mlx:
    sys.modules["mlx.core"] = old_mlx
  else:
    del sys.modules["mlx.core"]


def test_mlx_get_syntax():
  adapter = MLXAdapter()
  assert adapter.get_device_syntax("mps") == "mx.Device(mx.gpu)"
  assert adapter.get_device_check_syntax() == "mx.default_device() == mx.gpu"
  assert adapter.get_rng_split_syntax("r", "k") == "pass"
  assert adapter.get_serialization_imports() == ["import mlx.core as mx"]
  assert adapter.get_weight_conversion_imports() == ["import mlx.core as mx"]
