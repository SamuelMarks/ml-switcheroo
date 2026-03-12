import sys
import pytest
from unittest.mock import patch
import importlib


def test_import_errors_frameworks():
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

    from ml_switcheroo.enums import SemanticTier

    import ml_switcheroo.frameworks.jax

    a1 = ml_switcheroo.frameworks.jax.JaxCoreAdapter()
    a1.convert([1, 2, 3])
    try:
      a1.collect_api(SemanticTier.NEURAL)
    except:
      pass
    try:
      a1.collect_api(SemanticTier.NEURAL_OPS)
    except:
      pass

    import ml_switcheroo.frameworks.torch

    a2 = ml_switcheroo.frameworks.torch.TorchAdapter()
    a2.convert([1, 2, 3])
    try:
      a2.collect_api(SemanticTier.NEURAL)
    except:
      pass

    import ml_switcheroo.frameworks.tensorflow

    a3 = ml_switcheroo.frameworks.tensorflow.TensorFlowAdapter()
    a3.convert([1, 2, 3])
    try:
      a3.collect_api(SemanticTier.NEURAL)
    except:
      pass
    try:
      a3.collect_api(SemanticTier.NEURAL_OPS)
    except:
      pass
    try:
      a3.get_weight_load_code("a")
    except:
      pass

    import ml_switcheroo.frameworks.mlx

    a4 = ml_switcheroo.frameworks.mlx.MLXAdapter()
    a4.convert([1, 2, 3])
    try:
      a4.collect_api(SemanticTier.NEURAL)
    except:
      pass
    try:
      a4.collect_api(SemanticTier.NEURAL_OPS)
    except:
      pass

    import ml_switcheroo.frameworks.paxml

    a5 = ml_switcheroo.frameworks.paxml.PaxmlAdapter()
    a5.convert([1, 2, 3])
    try:
      a5.collect_api(SemanticTier.NEURAL)
    except:
      pass
