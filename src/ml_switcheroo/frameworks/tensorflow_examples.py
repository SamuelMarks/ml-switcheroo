"""TensorFlow Example Snippets."""

from typing import Dict


def get_tf_tiered_examples() -> Dict[str, str]:
  """Returns example snippets for each semantic tier."""
  return {
    "tier1_math": """import tensorflow as tf

def math_ops(x, y):
  # Tier 1: Core TensorFlow Math
  a = tf.abs(x)
  b = tf.math.add(a, y)
  return tf.math.reduce_mean(b)
""",
    "tier2_neural": """import tensorflow as tf

class Model(tf.Module):
  # Tier 2: Low-level TF Module (Not Keras)
  def __init__(self, in_features, out_features):
    super().__init__()
    self.w = tf.Variable(tf.random.normal([in_features, out_features]))
    self.b = tf.Variable(tf.zeros([out_features]))

  def __call__(self, x):
    return tf.matmul(x, self.w) + self.b
""",
    "tier3_extras": """import tensorflow as tf

def data_pipeline(tensors, batch_size=32):
  # Tier 3: tf.data Input Pipeline
  dataset = tf.data.Dataset.from_tensor_slices(tensors)
  loader = dataset.shuffle(1024).batch(batch_size)
  return loader
""",
    "tier4_qwen3-vl": """import tensorflow as tf

class Qwen3VLVisionConfig:
    in_channels: int = 3
    hidden_size: int = 1280
    temporal_patch_size: int = 2
    patch_size: int = 14

class Qwen3VLPatchEmbed(tf.keras.layers.Layer):
    '''3D Convolutional patch embedding for vision input.'''
    def __init__(self, config: Qwen3VLVisionConfig):
        super().__init__()
        self.config = config
        kernel = (config.temporal_patch_size, config.patch_size, config.patch_size)
        self.proj = tf.keras.layers.Conv3D(
            filters=config.hidden_size,
            kernel_size=kernel,
            strides=kernel,
            padding="valid",
            use_bias=True,
        )

    def call(self, hidden_states):
        cfg = self.config
        seq_len = tf.shape(hidden_states)[0]

        hidden_states = tf.reshape(
            hidden_states,
            [seq_len, cfg.in_channels, cfg.temporal_patch_size, cfg.patch_size, cfg.patch_size]
        )
        hidden_states = tf.transpose(hidden_states, perm=[0, 2, 3, 4, 1])

        out = self.proj(hidden_states)
        return tf.reshape(out, [seq_len, cfg.hidden_size])
""",
  }
