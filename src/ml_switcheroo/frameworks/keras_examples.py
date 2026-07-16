"""Keras Example Snippets."""

from typing import Dict


def get_keras_tiered_examples() -> Dict[str, str]:
  """Returns example snippets for each semantic tier."""
  return {
    "tier1_math": """import keras

def math_ops(x, y):
  # Tier 1: Keras Core Operations
  a = keras.ops.abs(x)
  b = keras.ops.add(a, y)
  return keras.ops.mean(b)
""",
    "tier2_neural_sequential": """import keras
from keras import layers

def build_model(input_shape):
  # Tier 2: Keras Sequential
  model = keras.Sequential([
      layers.Dense(64, activation='relu', input_shape=input_shape),
      layers.Dense(10, activation='softmax')
  ])
  return model
""",
    "tier3_extras_rng": """import keras

def generate_noise(shape):
  # Tier 3: Random Number Generation
  seed = keras.random.SeedGenerator(42)
  return keras.random.normal(shape, seed=seed)
""",
    "tier4_qwen3-vl": """import keras

class Qwen3VLVisionConfig:
    in_channels: int = 3
    hidden_size: int = 1280
    temporal_patch_size: int = 2
    patch_size: int = 14

class Qwen3VLPatchEmbed(keras.layers.Layer):
    '''3D Convolutional patch embedding for vision input.'''
    def __init__(self, config: Qwen3VLVisionConfig):
        super().__init__()
        self.config = config
        kernel = (config.temporal_patch_size, config.patch_size, config.patch_size)
        self.proj = keras.layers.Conv3D(
            filters=config.hidden_size,
            kernel_size=kernel,
            strides=kernel,
            padding="valid",
            use_bias=True,
        )

    def call(self, hidden_states):
        cfg = self.config
        seq_len = keras.ops.shape(hidden_states)[0]

        hidden_states = keras.ops.reshape(
            hidden_states,
            [seq_len, cfg.in_channels, cfg.temporal_patch_size, cfg.patch_size, cfg.patch_size]
        )
        hidden_states = keras.ops.transpose(hidden_states, axes=[0, 2, 3, 4, 1])

        out = self.proj(hidden_states)
        return keras.ops.reshape(out, [seq_len, cfg.hidden_size])
""",
  }
