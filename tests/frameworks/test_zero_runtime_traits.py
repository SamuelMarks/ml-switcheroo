"""Unit tests for zero-* runtime traits on framework adapters."""

from ml_switcheroo.frameworks.flax_nnx import FlaxNNXAdapter
from ml_switcheroo.frameworks.jax import JaxCoreAdapter
from ml_switcheroo.frameworks.keras import KerasAdapter
from ml_switcheroo.frameworks.mlx import MLXAdapter
from ml_switcheroo.frameworks.paxml import PaxmlAdapter
from ml_switcheroo.frameworks.tensorflow import TensorFlowAdapter
from ml_switcheroo.frameworks.torch import TorchAdapter


def test_torch_zero_runtime_traits() -> None:
  """Verify zero-* runtime traits on TorchAdapter."""
  traits = TorchAdapter().structural_traits
  assert traits.zero_runtime_package == "zero_torch"
  assert traits.zero_module_aliases["torch"] == "zero_torch"
  assert traits.zero_module_aliases["torch.nn"] == "zero_torch.nn"
  assert traits.zero_module_aliases["torch.nn.functional"] == "zero_torch.nn.functional"
  assert traits.zero_module_aliases["torch.optim"] == "zero_torch.optim"
  assert traits.zero_module_aliases["torch.autograd"] == "zero_torch.autograd"
  assert traits.zero_module_aliases["torch.fft"] == "zero_torch.fft"
  assert traits.zero_module_aliases["torch.linalg"] == "zero_torch.linalg"
  assert traits.zero_module_aliases["torch.special"] == "zero_torch.special"
  assert traits.autodiff_strategy == "imperative_backward"
  assert traits.shape_assertion_template == "assert tuple({var}.shape) == {shape}"
  assert traits.shape_assertion_import is None
  assert traits.lazy_layer_dim_arg_map == {
    "Linear": "in_features",
    "Conv1d": "in_channels",
    "Conv2d": "in_channels",
    "Conv3d": "in_channels",
  }


def test_jax_zero_runtime_traits() -> None:
  """Verify zero-* runtime traits on JaxCoreAdapter."""
  traits = JaxCoreAdapter().structural_traits
  assert traits.zero_runtime_package == "zero_jax"
  assert traits.zero_module_aliases["jax"] == "zero_jax"
  assert traits.zero_module_aliases["jax.numpy"] == "zero_jax.numpy"
  assert traits.zero_module_aliases["jax.lax"] == "zero_jax.lax"
  assert traits.zero_module_aliases["jax.nn"] == "zero_jax.nn"
  assert traits.zero_module_aliases["jax.random"] == "zero_jax.random"
  assert traits.zero_module_aliases["jax.sharding"] == "zero_jax.sharding"
  assert traits.zero_module_aliases["jax.scipy"] == "zero_jax.scipy"
  assert traits.zero_module_aliases["optax"] == "zero_optax"
  assert traits.zero_module_aliases["chex"] == "zero_chex"
  assert traits.zero_module_aliases["orbax"] == "zero_orbax"
  assert traits.zero_module_aliases["grain"] == "zero_grain"
  assert traits.autodiff_strategy == "functional_value_and_grad"
  assert traits.shape_assertion_template == "chex.assert_shape({var}, {shape})"
  assert traits.shape_assertion_import == "import chex"
  assert traits.lazy_layer_dim_arg_map == {}


def test_flax_nnx_zero_runtime_traits() -> None:
  """Verify zero-* runtime traits on FlaxNNXAdapter."""
  traits = FlaxNNXAdapter().structural_traits
  assert traits.zero_runtime_package == "zero_flax"
  assert traits.zero_module_aliases["flax"] == "zero_flax"
  assert traits.zero_module_aliases["flax.nnx"] == "zero_flax.nnx"
  assert traits.zero_module_aliases["flax.linen"] == "zero_flax.linen"
  assert traits.zero_module_aliases["jax"] == "zero_jax"
  assert traits.zero_module_aliases["jax.numpy"] == "zero_jax.numpy"
  assert traits.zero_module_aliases["optax"] == "zero_optax"
  assert traits.zero_module_aliases["chex"] == "zero_chex"
  assert traits.zero_module_aliases["orbax"] == "zero_orbax"
  assert traits.autodiff_strategy == "nnx_value_and_grad"
  assert traits.shape_assertion_template == "chex.assert_shape({var}, {shape})"
  assert traits.shape_assertion_import == "import chex"
  assert traits.lazy_layer_dim_arg_map == {
    "Linear": "in_features",
    "Conv1d": "in_features",
    "Conv2d": "in_features",
    "Conv3d": "in_features",
  }


def test_keras_zero_runtime_traits() -> None:
  """Verify zero-* runtime traits on KerasAdapter."""
  traits = KerasAdapter().structural_traits
  assert traits.zero_runtime_package == "zero_keras"
  assert traits.zero_module_aliases["keras"] == "zero_keras"
  assert traits.zero_module_aliases["keras.layers"] == "zero_keras.layers"
  assert traits.zero_module_aliases["keras.ops"] == "zero_keras.ops"
  assert traits.zero_module_aliases["keras.models"] == "zero_keras.models"
  assert traits.zero_module_aliases["keras.losses"] == "zero_keras.losses"
  assert traits.zero_module_aliases["keras.optimizers"] == "zero_keras.optimizers"
  assert traits.zero_module_aliases["keras.random"] == "zero_keras.random"
  assert traits.autodiff_strategy == "gradient_tape"
  assert traits.shape_assertion_template == "assert tuple({var}.shape) == {shape}"
  assert traits.shape_assertion_import is None


def test_tensorflow_zero_runtime_traits() -> None:
  """Verify zero-* runtime traits on TensorFlowAdapter."""
  traits = TensorFlowAdapter().structural_traits
  assert traits.zero_runtime_package == "zero_tensorflow"
  assert traits.zero_module_aliases["tensorflow"] == "zero_tensorflow"
  assert traits.zero_module_aliases["tensorflow.keras"] == "zero_keras"
  assert traits.zero_module_aliases["tensorflow.keras.layers"] == "zero_keras.layers"
  assert traits.zero_module_aliases["tensorflow.nn"] == "zero_tensorflow.nn"
  assert traits.zero_module_aliases["tensorflow.math"] == "zero_tensorflow.math"
  assert traits.zero_module_aliases["tensorflow.linalg"] == "zero_tensorflow.linalg"
  assert traits.autodiff_strategy == "gradient_tape"
  assert traits.shape_assertion_template == "assert tuple({var}.shape) == {shape}"
  assert traits.shape_assertion_import is None


def test_mlx_zero_runtime_traits() -> None:
  """Verify zero-* runtime traits on MLXAdapter."""
  traits = MLXAdapter().structural_traits
  assert traits.zero_runtime_package == "zero_mlx"
  assert traits.zero_module_aliases["mlx"] == "zero_mlx"
  assert traits.zero_module_aliases["mlx.core"] == "zero_mlx.core"
  assert traits.zero_module_aliases["mlx.nn"] == "zero_mlx.nn"
  assert traits.zero_module_aliases["mlx.optimizers"] == "zero_mlx.optimizers"
  assert traits.zero_module_aliases["mlx.fast"] == "zero_mlx.fast"
  assert traits.autodiff_strategy == "mlx_value_and_grad"
  assert traits.shape_assertion_template == "assert tuple({var}.shape) == {shape}"
  assert traits.shape_assertion_import is None
  assert traits.lazy_layer_dim_arg_map == {
    "Linear": "input_dims",
    "Conv1d": "in_channels",
    "Conv2d": "in_channels",
  }


def test_paxml_zero_runtime_traits() -> None:
  """Verify zero-* runtime traits on PaxmlAdapter."""
  traits = PaxmlAdapter().structural_traits
  assert traits.zero_runtime_package == "zero_pax"
  assert traits.zero_module_aliases["praxis"] == "zero_pax.praxis"
  assert traits.zero_module_aliases["praxis.layers"] == "zero_pax.praxis.layers"
  assert traits.zero_module_aliases["praxis.base_layer"] == "zero_pax.praxis.base_layer"
  assert traits.zero_module_aliases["praxis.pax_fiddle"] == "zero_pax.praxis.pax_fiddle"
  assert traits.zero_module_aliases["paxml"] == "zero_pax"
  assert traits.zero_module_aliases["jax"] == "zero_jax"
  assert traits.zero_module_aliases["jax.numpy"] == "zero_jax.numpy"
  assert traits.zero_module_aliases["chex"] == "zero_chex"
  assert traits.autodiff_strategy == "functional_value_and_grad"
  assert traits.shape_assertion_template == "chex.assert_shape({var}, {shape})"
  assert traits.shape_assertion_import == "import chex"
  assert traits.lazy_layer_dim_arg_map == {"Linear": "input_dims"}
