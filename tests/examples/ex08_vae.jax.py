"""Variational Autoencoder (VAE) mathematical and utility functions in JAX.

This module provides standard utility functions for building and training
Variational Autoencoders in JAX. It includes the reparameterization trick
for sampling from latent distributions, the element-wise binary cross-entropy
reconstruction loss, and the Kullback-Leibler (KL) divergence loss for normal
distributions.
"""

import jax.numpy as jnp
from jax import random


def sample_latent(mean, logvar, key):
  """VAE Reparameterization Trick.

  Samples from the latent Gaussian distribution N(mean, exp(logvar)) by scaling
  and shifting standard normal noise. This trick allows gradients to flow
  back through the stochastic node during backpropagation.

  Source: JAX Examples.

  Args:
      mean: A JAX array representing the mean of the latent distribution.
      logvar: A JAX array representing the log-variance of the latent
          distribution.
      key: A JAX PRNGKey used to generate the standard normal noise.

  Returns:
      A JAX array of the same shape as `mean` representing the sampled latent
      vectors.
  """
  std = jnp.exp(0.5 * logvar)
  eps = random.normal(key, logvar.shape)
  return mean + eps * std


def binary_cross_entropy(logits, x):
  """Computes the element-wise binary cross-entropy reconstruction loss.

  Calculates the negative log-likelihood of the target data `x` given the
  predicted probabilities or reconstructed values `logits`.

  Args:
      logits: A JAX array of predicted probabilities/reconstructed values,
          where each value is typically in the range [0, 1].
      x: A JAX array of target/ground-truth values, where each value is
          typically binary (0 or 1) or in the range [0, 1].

  Returns:
      A scalar JAX array representing the sum of the binary cross-entropy
      reconstruction losses over all elements.
  """
  # Element-wise binary cross entropy
  return -jnp.sum(x * jnp.log(logits) + (1 - x) * jnp.log(1 - logits))


def gaussian_kl(mean, logvar):
  """Computes the Kullback-Leibler (KL) divergence.

  Calculates the KL divergence between a parameterized diagonal Gaussian
  distribution N(mean, exp(logvar)) and a standard normal distribution N(0, I).

  Args:
      mean: A JAX array representing the mean of the Gaussian distribution.
      logvar: A JAX array representing the log-variance of the Gaussian
          distribution.

  Returns:
      A scalar JAX array representing the sum of the KL divergence losses
      over all elements.
  """
  # KL divergence between Gaussian and standard normal
  return -0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar))
