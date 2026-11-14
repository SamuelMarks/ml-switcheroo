"""Keras 3 golden reference for Weight-Normalized Dense projection."""

import keras


class WNModel:
  """Weight-normalized Dense projection in Keras 3."""

  def __init__(self, in_features: int, out_features: int) -> None:
    """Initialize Dense layer and scalar magnitude gain.

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
    """
    self.in_features = in_features
    self.dense = keras.layers.Dense(out_features)

  def call(self, x: keras.KerasTensor) -> keras.KerasTensor:
    """Compute Dense projection with L2-normalized weight vectors.

    Args:
        x: Input ``KerasTensor`` operand.

    Returns:
        Projected ``KerasTensor`` output.
    """
    out = self.dense(x)
    kernel = self.dense.kernel
    denom = keras.ops.sqrt(keras.ops.sum(keras.ops.square(kernel), axis=0, keepdims=True) + 1e-12)
    res: keras.KerasTensor = out / denom
    return res
