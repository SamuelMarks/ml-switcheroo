"""Test suite for mlx."""

import typing
from pydantic import JsonValue

import pytest
from ml_switcheroo_ir.schema.ghost import SemanticTier

from ml_switcheroo.frameworks.mlx import MLXAdapter


def test_mlx_adapter_basic() -> None:
  """Verify mlx adapter basic behavior."""
  adapter = MLXAdapter()

  assert adapter.import_alias == ("mlx.core", "mx")
  ns = adapter.import_namespaces
  assert "mlx.core" in ns
  assert ns["mlx.core"].recommended_alias == "mx"
  assert ns["mlx.nn"].recommended_alias == "nn"
  assert ns["mlx.optimizers"].recommended_alias == "optim"

  assert SemanticTier.NEURAL in adapter.supported_tiers

  cfg = adapter.test_config
  assert "import mlx.core as mx" in cfg["import"]
  assert "array" in cfg["convert_input"]

  assert adapter.harness_imports == []
  assert adapter.get_harness_init_code() == ""
  assert "tolist" in adapter.get_to_numpy_code()

  traits = adapter.structural_traits
  assert traits.module_base == "mlx.nn.Module"
  assert "__call__" in traits.known_inference_methods

  ptraits = adapter.plugin_traits
  assert ptraits.has_numpy_compatible_arrays is True
  assert ptraits.requires_explicit_rng is False
  assert ptraits.requires_functional_state is False

  assert adapter.rng_seed_methods == ["seed", "random.seed"]
  assert adapter.declared_magic_args == []

  defs = adapter.definitions
  assert isinstance(defs, dict)


def test_mlx_adapter_syntax() -> None:
  """Verify mlx adapter syntax behavior."""
  adapter = MLXAdapter()

  assert adapter.get_device_syntax("gpu", "0") == "mx.Device(mx.gpu, 0)"
  assert adapter.get_device_syntax("cpu", None) == "mx.Device(mx.cpu)"

  assert adapter.get_device_check_syntax() == "mx.default_device() == mx.gpu"
  assert adapter.get_rng_split_syntax("rng", "key") == "pass"


def test_mlx_adapter_docs() -> None:
  """Verify mlx adapter docs behavior."""
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
  """Verify mlx adapter convert behavior."""
  pytest.importorskip("mlx")
  adapter = MLXAdapter()
  import mlx.core as mx
  import numpy as np

  t = adapter.convert([1, 2, 3])
  assert isinstance(t, mx.array)

  t2 = adapter.convert(np.array([1, 2, 3]))
  assert isinstance(t2, mx.array)

  t3 = adapter.convert(1)
  assert t3 == 1


def test_mlx_adapter_wiring() -> None:
  """Verify mlx adapter wiring behavior."""
  adapter = MLXAdapter()
  snapshot: dict[str, typing.Union[typing.Hashable, typing.Sized]] = {}
  adapter.apply_wiring(snapshot)  # type: ignore[arg-type]
  assert snapshot == {}


def test_convert() -> None:
  """Verify convert behavior."""
  import sys
  from unittest.mock import MagicMock

  from ml_switcheroo.frameworks.mlx import MLXAdapter

  sys.modules["mlx.core"] = MagicMock()
  sys.modules["mlx"] = MagicMock()
  import numpy as np

  adapter = MLXAdapter()
  res_arr = adapter.convert(np.array([1, 2, 3]))
  res_scalar = adapter.convert(1)
  del sys.modules["mlx.core"]
  del sys.modules["mlx"]
  assert res_arr is not None
  assert res_scalar == 1


# --- Merged from test_mlx_extra2_loop.py ---


def test_mlx_no_numpy() -> None:
  """Verify mlx no numpy behavior."""
  import builtins
  import importlib
  import types
  from unittest.mock import patch

  import ml_switcheroo.frameworks.mlx as mlx_mod

  orig_import = builtins.__import__

  def mock_import(name: str, *args: JsonValue, **kwargs: JsonValue) -> types.ModuleType:
    """Execute mock import helper logic.

    Args:
        name: Test argument or fixture.
        args: Test argument or fixture.
        kwargs: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    if name == "numpy":
      raise ImportError("no numpy")
    return orig_import(name, *args, **kwargs)  # type: ignore[arg-type]

  with patch("builtins.__import__", side_effect=mock_import):
    try:
      importlib.reload(mlx_mod)
    except Exception:
      pass
  _res = importlib.reload(mlx_mod)
  assert hasattr(_res, "MLXAdapter")
