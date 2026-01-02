""" 
Tests for API Documentation URL Generation Protocol. 

Verifies that: 
1.  All adapters implement `get_doc_url`. 
2.  Generated URLs match expected patterns for each framework. 
3.  Fallback behaviors (None) are correct for intermediate reps. 
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

def test_stub_adapters_return_none(): 
  """Verify intermediate/DSL adapters return None for docs.""" 
  stubs = ["mlir", "tikz", "html", "latex_dsl"] 
  for key in stubs: 
    adapter = get_adapter(key) 
    if adapter: 
      assert adapter.get_doc_url("foo") is None