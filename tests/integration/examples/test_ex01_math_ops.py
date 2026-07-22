"""Test suite for the Ex01 Math Ops module."""

import pytest
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager

SOURCE_TORCH = "\nimport torch\n\ndef compute_loss(prediction, target):\n    diff = torch.abs(prediction - target)\n    loss = torch.mean(diff)\n    return loss\n"
EXPECTED_JAX = "\nimport jax.numpy as jnp\n\ndef compute_loss(prediction, target):\n    diff = jnp.abs(prediction - target)\n    loss = jnp.mean(diff)\n    return loss\n"
EXPECTED_NUMPY = "\nimport numpy as np\n\ndef compute_loss(prediction, target):\n    diff = np.abs(prediction - target)\n    loss = np.mean(diff)\n    return loss\n"
EXPECTED_TENSORFLOW = "\nimport tensorflow as tf\n\ndef compute_loss(prediction, target):\n    diff = tf.abs(prediction - target)\n    loss = tf.math.reduce_mean(diff)\n    return loss\n"
EXPECTED_MLX = "\nimport mlx.core as mx\n\ndef compute_loss(prediction, target):\n    diff = mx.abs(prediction - target)\n    loss = mx.mean(diff)\n    return loss\n"
EXPECTED_KERAS = "\nimport keras\nimport numpy as np\n\ndef compute_loss(prediction, target):\n    diff = keras.ops.abs(prediction - target)\n    loss = keras.ops.mean(diff)\n    return loss\n"


@pytest.fixture(scope="module")
def semantics():
  """Helper to semantics."""
  mgr = SemanticsManager()
  abs_def = {
    "std_args": ["x"],
    "variants": {
      "torch": {"api": "torch.abs"},
      "jax": {"api": "jax.numpy.abs"},
      "numpy": {"api": "numpy.abs"},
      "tensorflow": {"api": "tf.abs"},
      "mlx": {"api": "mlx.core.abs"},
      "keras": {"api": "keras.ops.abs"},
    },
  }
  mean_def = {
    "std_args": ["x"],
    "variants": {
      "torch": {"api": "torch.mean"},
      "jax": {"api": "jax.numpy.mean"},
      "numpy": {"api": "numpy.mean"},
      "tensorflow": {"api": "tf.math.reduce_mean"},
      "mlx": {"api": "mlx.core.mean"},
      "keras": {"api": "keras.ops.mean"},
    },
  }
  mgr.update_definition("Abs", abs_def)
  mgr.update_definition("Mean", mean_def)
  return mgr


@pytest.mark.parametrize(
  "target_fw, expected_string",
  [("jax", "jnp.abs"), ("numpy", "np.abs"), ("tensorflow", "tf.abs"), ("mlx", "mx.abs"), ("keras", "keras.ops.abs")],
)
def test_ex01_math_transpilation(semantics, target_fw, expected_string):
  """Verifies the behavior of ex01 math transpilation."""
  config = RuntimeConfig(source_framework="torch", target_framework=target_fw, strict_mode=True)
  engine = ASTEngine(semantics=semantics, config=config)
  result = engine.run(SOURCE_TORCH)
  assert result.success, f"Failed converting to {target_fw}: {result.errors}"
  assert expected_string in result.code
  assert "compute_loss" in result.code
  assert "prediction - target" in result.code
  if target_fw == "jax":
    assert "import jax.numpy as jnp" in result.code
  elif target_fw == "numpy":
    assert "import numpy as np" in result.code
  elif target_fw == "mlx":
    assert "import mlx.core as mx" in result.code
