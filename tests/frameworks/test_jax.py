"""Test suite for the Jax module."""

from ml_switcheroo.frameworks.jax import JaxCoreAdapter
from ml_switcheroo.frameworks.base import InitMode


def test_jax_adapter_init():
  """Verifies the behavior of JAX adapter initialization."""
  adapter = JaxCoreAdapter()
  assert adapter.display_name == "JAX (no framework)"
  assert adapter.inherits_from is None
  assert adapter.ui_priority == 10


def test_jax_import_alias():
  """Verifies the behavior of JAX import alias."""
  adapter = JaxCoreAdapter()
  assert adapter.import_alias == ("jax.numpy", "jnp")


def test_jax_import_namespaces():
  """Verifies the behavior of JAX import namespaces."""
  adapter = JaxCoreAdapter()
  ns = adapter.import_namespaces
  assert "jax.numpy" in ns
  assert ns["jax.numpy"].recommended_alias == "jnp"
  assert "optax" in ns


def test_jax_test_config():
  """Verifies the behavior of JAX test configuration."""
  adapter = JaxCoreAdapter()
  config = adapter.test_config
  assert "import jax.numpy as jnp" in config["import"]


def test_jax_harness_imports():
  """Verifies the behavior of JAX harness imports."""
  adapter = JaxCoreAdapter()
  assert "import jax" in adapter.harness_imports
  assert "import jax.random" in adapter.harness_imports


def test_jax_harness_init_code():
  """Verifies the behavior of JAX harness initialization code."""
  adapter = JaxCoreAdapter()
  code = adapter.get_harness_init_code()
  assert "def _make_jax_key(seed):" in code
  assert "jax.random.PRNGKey(seed)" in code


def test_jax_declared_magic_args():
  """Verifies the behavior of JAX declared magic arguments."""
  adapter = JaxCoreAdapter()
  assert adapter.declared_magic_args == ["key"]


def test_jax_structural_traits():
  """Verifies the behavior of JAX structural traits."""
  adapter = JaxCoreAdapter()
  traits = adapter.structural_traits
  assert traits.module_base is None
  assert traits.forward_method == "__call__"
  assert not traits.requires_super_init


def test_jax_rng_seed_methods():
  """Verifies the behavior of JAX rng seed methods."""
  adapter = JaxCoreAdapter()
  assert adapter.rng_seed_methods == []


def test_jax_definitions(monkeypatch):
  """Verifies the behavior of JAX definitions."""
  adapter = JaxCoreAdapter()
  defs = adapter.definitions
  assert isinstance(defs, dict)


def test_jax_convert_no_array():
  """Verifies the behavior of JAX convert no array."""
  adapter = JaxCoreAdapter()
  assert adapter.convert("test") == "test"


def test_jax_convert_list():
  """Verifies the behavior of JAX convert list."""
  adapter = JaxCoreAdapter()
  res = adapter.convert([1, 2, 3])
  assert res is not None


def test_jax_apply_wiring():
  """Verifies the behavior of JAX apply wiring."""
  adapter = JaxCoreAdapter()
  snapshot = {}
  adapter.apply_wiring(snapshot)
  assert "mappings" in snapshot
  assert "templates" in snapshot


def test_jax_tiered_examples():
  """Verifies the behavior of JAX tiered examples."""
  adapter = JaxCoreAdapter()
  examples = adapter.get_tiered_examples()
  assert "tier1_math" in examples
  assert "tier2_neural" in examples
  assert "tier4_qwen3-vl" in examples
  assert "tier3_extras" in examples


def test_jax_doc_url():
  """Verifies the behavior of JAX documentation URL."""
  adapter = JaxCoreAdapter()
  url = adapter.get_doc_url("jax.numpy.abs")
  assert url == "https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.abs.html"


def test_jax_init_live_mode(monkeypatch):
  """Verifies the behavior of JAX initialization live mode."""
  monkeypatch.setattr("ml_switcheroo.frameworks.jax.jax", True)
  adapter = JaxCoreAdapter()
  assert adapter._mode == InitMode.LIVE
