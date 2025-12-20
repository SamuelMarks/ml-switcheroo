import keras
from keras import layers, ops


def build_model(input_shape, num_classes):
  """
  Keras Functional API Model.
  Source: Keras Examples.
  """
  inputs = keras.Input(shape=input_shape)
  x = layers.Conv2D(32, kernel_size=(3, 3), activation="relu")(inputs)
  x = layers.MaxPooling2D(pool_size=(2, 2))(x)
  x = layers.Flatten()(x)
  x = layers.Dropout(0.5)(x)
  outputs = layers.Dense(num_classes, activation="softmax")(x)
  return keras.Model(inputs, outputs)


def custom_loss(y_true, y_pred):
  # Using backend-agnostic ops
  return ops.mean(ops.square(y_true - y_pred))
