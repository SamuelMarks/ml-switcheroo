"""Docstring."""

import typing

import pytest
from ml_switcheroo_ir.schema.ghost import SemanticTier

from ml_switcheroo.frameworks.mlx import MLXAdapter


def test_mlx_adapter_basic() -> None:
  """Docstring."""
  adapter = MLXAdapter()

  assert adapter.import_alias == ("mlx.core", "mx")
  ns: typing.Any = adapter.import_namespaces
  assert "mlx.core" in ns
  assert ns["mlx.core"].recommended_alias == "mx"
  assert ns["mlx.nn"].recommended_alias == "nn"
  assert ns["mlx.optimizers"].recommended_alias == "optim"

  assert SemanticTier.NEURAL in adapter.supported_tiers

  cfg: dict[str, typing.Any] = adapter.test_config
  assert "import mlx.core as mx" in cfg["import"]
  assert "array" in cfg["convert_input"]

  assert adapter.harness_imports == []
  assert adapter.get_harness_init_code() == ""
  assert "tolist" in adapter.get_to_numpy_code()

  traits: typing.Any = adapter.structural_traits
  assert traits.module_base == "mlx.nn.Module"
  assert "__call__" in traits.known_inference_methods

  ptraits: typing.Any = adapter.plugin_traits
  assert ptraits.has_numpy_compatible_arrays is True
  assert ptraits.requires_explicit_rng is False
  assert ptraits.requires_functional_state is False

  assert adapter.rng_seed_methods == ["seed", "random.seed"]
  assert adapter.declared_magic_args == []

  defs: typing.Any = adapter.definitions
  assert isinstance(defs, dict)


def test_mlx_adapter_syntax() -> None:
  """Docstring."""
  adapter = MLXAdapter()

  assert adapter.get_device_syntax("gpu", "0") == "mx.Device(mx.gpu, 0)"
  assert adapter.get_device_syntax("cpu", None) == "mx.Device(mx.cpu)"

  assert adapter.get_device_check_syntax() == "mx.default_device() == mx.gpu"
  assert adapter.get_rng_split_syntax("rng", "key") == "pass"


def test_mlx_adapter_docs() -> None:
  """Docstring."""
  adapter = MLXAdapter()
  url: typing.Optional[str] = adapter.get_doc_url("mlx.core.abs")
  assert url is not None
  assert "ml-explore.github.io" in url

  url2: typing.Optional[str] = adapter.get_doc_url("unknown")
  assert url2 is not None
  assert "unknown.html" in url2

  examples: dict[str, str] = adapter.get_tiered_examples()
  assert len(examples) > 0


def test_mlx_adapter_convert() -> None:
  """Docstring."""
  pytest.importorskip("mlx")
  adapter = MLXAdapter()
  import mlx.core as mx  # type: ignore
  import numpy as np

  t: typing.Any = adapter.convert([1, 2, 3])
  assert isinstance(t, mx.array)

  t2: typing.Any = adapter.convert(np.array([1, 2, 3]))
  assert isinstance(t2, mx.array)

  t3: typing.Any = adapter.convert(1)
  assert t3 == 1


def test_mlx_adapter_wiring() -> None:
  """Docstring."""
  adapter = MLXAdapter()
  adapter.apply_wiring({})


def test_convert() -> None:
  """Docstring."""
  import sys
  from unittest.mock import MagicMock

  from ml_switcheroo.frameworks.mlx import MLXAdapter

  sys.modules["mlx.core"] = MagicMock()
  sys.modules["mlx"] = MagicMock()
  import numpy as np

  adapter = MLXAdapter()
  adapter.convert(np.array([1, 2, 3]))
  adapter.convert(1)
  del sys.modules["mlx.core"]
  del sys.modules["mlx"]


# --- Merged from test_mlx_extra2_loop.py ---


def test_mlx_no_numpy() -> None:
  """Docstring."""
  import builtins
  import importlib
  from unittest.mock import patch

  import ml_switcheroo.frameworks.mlx as mlx_mod

  orig_import = builtins.__import__

  def mock_import(name, *args, **kwargs):
    """Docstring."""
    if name == "numpy":
      raise ImportError("no numpy")
    return orig_import(name, *args, **kwargs)

  with patch("builtins.__import__", side_effect=mock_import):
    try:
      importlib.reload(mlx_mod)
    except Exception:
      pass
  importlib.reload(mlx_mod)
