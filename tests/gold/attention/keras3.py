"""Module docstring."""

import keras


class AttentionModel(keras.Model):
  """Class docstring."""

  def __init__(self, embed_dim: int, num_heads: int):
    """Function docstring."""
    super().__init__()
    # <SWITCHEROO_FAILED_TO_TRANS>
    # Keras names arguments differently: key_dim
    self.mha = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)
    # </SWITCHEROO_FAILED_TO_TRANS>

  def call(self, query, value, key=None):
    """Function docstring."""
    # Keras signature is usually (query, value, key)
    return self.mha(query, value, key=key)
