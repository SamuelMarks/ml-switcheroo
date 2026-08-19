"""Tests for Keras and TF convolutional layers mappings."""

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.semantics.manager import SemanticsManager


def test_conv2d_keras_positional():
  """Test positional mapping for Conv2d to Keras."""
  code = "import torch.nn as nn\nself.conv = nn.Conv2d(1, 32, 3)"
  config = RuntimeConfig(source_framework="torch", target_framework="keras")
  semantics = SemanticsManager()
  engine = ASTEngine(semantics, config)
  result = engine.run(code)

  assert "layers.Conv2D(32, 3)" in result.code


def test_conv2d_tensorflow_positional():
  """Test positional mapping for Conv2d to TensorFlow."""
  code = "import torch.nn as nn\nself.conv = nn.Conv2d(1, 32, 3)"
  config = RuntimeConfig(source_framework="torch", target_framework="tensorflow")
  semantics = SemanticsManager()
  engine = ASTEngine(semantics, config)
  result = engine.run(code)

  assert "tf.keras.layers.Conv2D(32, 3)" in result.code


def test_conv_transpose2d_keras_positional():
  """Test positional mapping for ConvTranspose2d to Keras."""
  code = "import torch.nn as nn\nself.conv = nn.ConvTranspose2d(1, 32, 3)"
  config = RuntimeConfig(source_framework="torch", target_framework="keras")
  semantics = SemanticsManager()
  engine = ASTEngine(semantics, config)
  result = engine.run(code)

  assert "layers.Conv2DTranspose(32, 3" in result.code


def test_conv1d_keras_positional():
  """Test positional mapping for Conv1d to Keras."""
  code = "import torch.nn as nn\nself.conv = nn.Conv1d(1, 32, 3)"
  config = RuntimeConfig(source_framework="torch", target_framework="keras")
  semantics = SemanticsManager()
  engine = ASTEngine(semantics, config)
  result = engine.run(code)

  assert "layers.Conv1D(32, 3" in result.code


def test_conv3d_tensorflow_positional():
  """Test positional mapping for Conv3d to TensorFlow."""
  code = "import torch.nn as nn\nself.conv = nn.Conv3d(1, 32, 3)"
  config = RuntimeConfig(source_framework="torch", target_framework="tensorflow")
  semantics = SemanticsManager()
  engine = ASTEngine(semantics, config)
  result = engine.run(code)

  assert "tf.keras.layers.Convolution3D(32, 3" in result.code
