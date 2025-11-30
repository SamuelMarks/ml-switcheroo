from flax import nnx


class SimplePerceptron(nnx.Module):
  """
  Basic Single-Layer Perceptron in Flax NNX.
  Semantic pivot:
    - flax.nnx.Module -> torch.nn.Module
    - flax.nnx.Linear -> torch.nn.Linear
    - __call__ -> forward
  """

  def __init__(self, in_features, out_features, rngs: nnx.Rngs):
    self.layer = nnx.Linear(in_features, out_features, rngs=rngs)

  def __call__(self, x):
    return self.layer(x)
