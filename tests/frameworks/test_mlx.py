"""Test suite for the Mlx module."""

from ml_switcheroo.frameworks.mlx import MLXAdapter


def test_mlx_adapter_init():
  """Verifies the behavior of MLX adapter initialization."""
  adapter = MLXAdapter()
  assert adapter.display_name == "Apple MLX"
  assert adapter.inherits_from is None
  assert adapter.ui_priority == 50


def test_mlx_import_alias():
  """Verifies the behavior of MLX import alias."""
  adapter = MLXAdapter()
  assert adapter.import_alias == ("mlx.core", "mx")


def test_mlx_import_namespaces():
  """Verifies the behavior of MLX import namespaces."""
  adapter = MLXAdapter()
  ns = adapter.import_namespaces
  assert "mlx.core" in ns
  assert "mlx.nn" in ns
  assert "mlx.optimizers" in ns


def test_mlx_test_config():
  """Verifies the behavior of MLX test configuration."""
  adapter = MLXAdapter()
  config = adapter.test_config
  assert "import mlx.core as mx" in config["import"]


def test_mlx_harness_imports():
  """Verifies the behavior of MLX harness imports."""
  adapter = MLXAdapter()
  assert adapter.harness_imports == []
  assert adapter.get_harness_init_code() == ""


def test_mlx_get_to_numpy_code():
  """Verifies the behavior of MLX get to NumPy code."""
  adapter = MLXAdapter()
  assert "hasattr(obj, 'tolist')" in adapter.get_to_numpy_code()


def test_mlx_supported_tiers():
  """Verifies the behavior of MLX supported tiers."""
  adapter = MLXAdapter()
  assert len(adapter.supported_tiers) == 3


def test_mlx_declared_magic_args():
  """Verifies the behavior of MLX declared magic arguments."""
  adapter = MLXAdapter()
  assert adapter.declared_magic_args == []


def test_mlx_structural_traits():
  """Verifies the behavior of MLX structural traits."""
  adapter = MLXAdapter()
  traits = adapter.structural_traits
  assert traits.module_base == "mlx.nn.Module"


def test_mlx_definitions():
  """Verifies the behavior of MLX definitions."""
  adapter = MLXAdapter()
  defs = adapter.definitions
  assert isinstance(defs, dict)


def test_mlx_rng_seed_methods():
  """Verifies the behavior of MLX rng seed methods."""
  adapter = MLXAdapter()
  assert "seed" in adapter.rng_seed_methods


def test_mlx_convert():
  """Verifies the behavior of MLX convert."""
  adapter = MLXAdapter()
  assert adapter.convert("test") == "test"


def test_mlx_tiered_examples():
  """Verifies the behavior of MLX tiered examples."""
  adapter = MLXAdapter()
  examples = adapter.get_tiered_examples()
  assert "tier1_math" in examples
  assert "tier2_neural" in examples


def test_mlx_device_syntax():
  """Verifies the behavior of MLX device syntax."""
  adapter = MLXAdapter()
  assert "mx.Device(mx.gpu)" == adapter.get_device_syntax("cuda")
  assert "mx.Device(mx.cpu)" == adapter.get_device_syntax("cpu")
  assert "mx.Device(mx.gpu, 1)" == adapter.get_device_syntax("cuda", "1")


def test_mlx_device_check_syntax():
  """Verifies the behavior of MLX device check syntax."""
  adapter = MLXAdapter()
  assert "mx.default_device() == mx.gpu" in adapter.get_device_check_syntax()


def test_mlx_apply_wiring():
  """Verifies the behavior of MLX apply wiring."""
  adapter = MLXAdapter()
  snapshot = {}
  adapter.apply_wiring(snapshot)
  assert snapshot == {}


def test_mlx_doc_url():
  """Verifies the behavior of MLX documentation URL."""
  adapter = MLXAdapter()
  url = adapter.get_doc_url("mlx.core.abs")
  assert "mlx.core.abs.html" in url
