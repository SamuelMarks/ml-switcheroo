import jax.numpy as jnp
from jax import random


def sample_latent(mean, logvar, key):
  """
  VAE Reparameterization Trick.
  Source: JAX Examples.
  """
  std = jnp.exp(0.5 * logvar)
  eps = random.normal(key, logvar.shape)
  return mean + eps * std


def binary_cross_entropy(logits, x):
  # Element-wise binary cross entropy
  return -jnp.sum(x * jnp.log(logits) + (1 - x) * jnp.log(1 - logits))


def gaussian_kl(mean, logvar):
  # KL divergence between Gaussian and standard normal
  return -0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar))
