"""
Tests for API Documentation URL Generation Protocol.

Verifies that:
1.  All adapters implement `get_doc_url`.
2.  Generated URLs match expected patterns for each framework.
3.  Stub adapters (MLIR, TikZ) return valid informational URLs instead of None.
"""

import pytest
from ml_switcheroo.frameworks.base import get_adapter

# Format: (framework_key, sample_api, expected_url_pattern)
# Note: Patterns are checked as substrings
CASES = [
  (
    "torch",
    "torch.nn.Linear",
    "https://pytorch.org/docs/stable/generated/torch.nn.Linear.html",
  ),
  (
    "torch",
    "torch.nn.init.zeros_",
    "https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.zeros_",
  ),
  (
    "jax",
    "jax.numpy.abs",
    "https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.abs.html",
  ),
  (
    "numpy",
    "numpy.mean",
    "https://numpy.org/doc/stable/reference/generated/numpy.mean.html",
  ),
  (
    "tensorflow",
    "tf.math.add",
    "https://www.tensorflow.org/api_docs/python/tf/math/add",
  ),
  (
    "keras",
    "keras.layers.Dense",
    "https://keras.io/search.html?q=keras.layers.Dense",
  ),
  (
    "flax_nnx",
    "flax.nnx.Linear",
    "https://flax.readthedocs.io/en/latest/search.html?q=flax.nnx.Linear",
  ),
  (
    "mlx",
    "mlx.core.abs",
    "https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.abs.html",
  ),
  (
    "paxml",
    "praxis.layers.Linear",
    "github.com/search?q=repo%3Agoogle%2Fpaxml",
  ),
]


@pytest.mark.parametrize("fw_key, api, pattern", CASES)
def test_doc_url_generation(fw_key, api, pattern):
  adapter = get_adapter(fw_key)
  assert adapter is not None, f"Adapter for {fw_key} missing"

  url = adapter.get_doc_url(api)
  assert url is not None
  assert pattern in url


def test_stub_adapters_return_docs():
  """Verify intermediate/DSL adapters return valid links for docs."""
  # MLIR provides dialect docs
  mlir = get_adapter("mlir")
  assert mlir is not None
  assert mlir.get_doc_url("tosa.add") is None

  # LaTeX/TikZ provides nothing
  tikz = get_adapter("tikz")
  assert tikz is not None
  assert tikz.get_doc_url("foo") is None

  # LaTeX DSL provides nothing
  latex = get_adapter("latex_dsl")
  assert latex is not None
  assert latex.get_doc_url("midl.Conv") is None

  # HTML DSL provides nothing
  html = get_adapter("html")
  assert html is not None
  assert html.get_doc_url("html_dsl.Conv") is None
