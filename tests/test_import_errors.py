"""Test suite for the Import Errors module."""

import importlib
from unittest.mock import patch

import warnings

try:
  with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import flax  # noqa: F401
    import jax  # noqa: F401
    import tensorflow  # noqa: F401
    import torch  # noqa: F401
except ImportError:
  pass


def test_import_errors_frameworks() -> None:
  """Verifies the behavior of import errors frameworks."""
  with patch.dict(
    "sys.modules",
    {
      "jax": None,
      "jax.numpy": None,
      "flax": None,
      "flax.nnx": None,
      "mlx": None,
      "mlx.core": None,
      "mlx.nn": None,
      "paxml": None,
      "praxis": None,
      "tensorflow": None,
      "torch": None,
      "torch.nn": None,
      "optax": None,
      "torch.utils.data": None,
    },
  ):
    import ml_switcheroo.frameworks.flax_nnx

    importlib.reload(ml_switcheroo.frameworks.flax_nnx)
    import ml_switcheroo.frameworks.jax

    importlib.reload(ml_switcheroo.frameworks.jax)
    import ml_switcheroo.frameworks.mlx

    importlib.reload(ml_switcheroo.frameworks.mlx)
    import ml_switcheroo.frameworks.paxml

    importlib.reload(ml_switcheroo.frameworks.paxml)
    import ml_switcheroo.frameworks.tensorflow

    importlib.reload(ml_switcheroo.frameworks.tensorflow)
    import ml_switcheroo.frameworks.torch

    importlib.reload(ml_switcheroo.frameworks.torch)

    import ml_switcheroo.frameworks.jax

    a1: ml_switcheroo.frameworks.jax.JaxCoreAdapter = ml_switcheroo.frameworks.jax.JaxCoreAdapter()
    a1.convert([1, 2, 3])
  assert True
