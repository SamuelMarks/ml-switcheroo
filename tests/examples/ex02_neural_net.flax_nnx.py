"""Example of a simple neural network built using Flax NNX.

This module defines a basic single-layer perceptron model which is utilized for
demonstrating and testing translation capabilities from Flax NNX to PyTorch/JAX structures.
"""

from flax import nnx


class SimplePerceptron(nnx.Module):
  """Basic Single-Layer Perceptron in Flax NNX.

  Semantic pivot:
    - flax.nnx.Module -> torch.nn.Module
    - flax.nnx.Linear -> torch.nn.Linear
    - __call__ -> forward.
  """

  def __init__(self, in_features, out_features, rngs: nnx.Rngs):
    """Initializes the SimplePerceptron layer with a linear transformation.

    Args:
      in_features (int): The number of input features/dimensions.
      out_features (int): The number of output features/dimensions.
      rngs (flax.nnx.Rngs): The random number generator key collection for parameter initialization.
    """
    self.layer = nnx.Linear(in_features, out_features, rngs=rngs)

  def __call__(self, x):
    """Performs a forward pass of the neural network layer.

    Args:
      x (jax.Array): The input tensor to transform.

    Returns:
      jax.Array: The output tensor resulting from the linear transformation.
    """
    return self.layer(x)
