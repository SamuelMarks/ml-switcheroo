"""Test suite for the Flax Nnx module."""

from ml_switcheroo.frameworks.flax_nnx import FlaxNNXAdapter
from ml_switcheroo_ir.schema.ghost import SemanticTier


def test_flax_nnx_adapter_init():
  """Verifies the behavior of Flax NNX adapter initialization."""
  adapter = FlaxNNXAdapter()
  assert adapter.display_name == "Flax NNX"
  assert adapter.inherits_from == "jax"
  assert adapter.ui_priority == 15


def test_flax_nnx_import_alias():
  """Verifies the behavior of Flax NNX import alias."""
  adapter = FlaxNNXAdapter()
  assert adapter.import_alias == ("flax.nnx", "nnx")


def test_flax_nnx_import_namespaces():
  """Verifies the behavior of Flax NNX import namespaces."""
  adapter = FlaxNNXAdapter()
  namespaces = adapter.import_namespaces
  assert "flax.nnx" in namespaces
  assert namespaces["flax.nnx"].recommended_alias == "nnx"


def test_flax_nnx_test_config():
  """Verifies the behavior of Flax NNX test configuration."""
  adapter = FlaxNNXAdapter()
  config = adapter.test_config
  assert "import flax.nnx as nnx" in config["import"]
  assert "jax.jit" in config["jit_template"]


def test_flax_nnx_harness_imports():
  """Verifies the behavior of Flax NNX harness imports."""
  adapter = FlaxNNXAdapter()
  assert "from flax import nnx" in adapter.harness_imports


def test_flax_nnx_harness_init_code():
  """Verifies the behavior of Flax NNX harness initialization code."""
  adapter = FlaxNNXAdapter()
  code = adapter.get_harness_init_code()
  assert "def _make_flax_rngs(seed):" in code
  assert "nnx.Rngs(seed)" in code


def test_flax_nnx_supported_tiers():
  """Verifies the behavior of Flax NNX supported tiers."""
  adapter = FlaxNNXAdapter()
  tiers = adapter.supported_tiers
  assert SemanticTier.ARRAY_API in tiers


def test_flax_nnx_declared_magic_args():
  """Verifies the behavior of Flax NNX declared magic arguments."""
  adapter = FlaxNNXAdapter()
  assert "rngs" in adapter.declared_magic_args


def test_flax_nnx_structural_traits():
  """Verifies the behavior of Flax NNX structural traits."""
  adapter = FlaxNNXAdapter()
  traits = adapter.structural_traits
  assert traits.module_base == "flax.nnx.Module"
  assert traits.forward_method == "__call__"
  assert not traits.requires_super_init


def test_flax_nnx_definitions():
  """Verifies the behavior of Flax NNX definitions."""
  adapter = FlaxNNXAdapter()
  defs = adapter.definitions
  assert "Module" in defs
  assert defs["Module"].api == "flax.nnx.Module"
  assert "relu" in defs
  assert defs["relu"].api == "flax.nnx.relu"


def test_flax_nnx_apply_wiring():
  """Verifies the behavior of Flax NNX apply wiring."""
  adapter = FlaxNNXAdapter()
  snapshot = {}
  adapter.apply_wiring(snapshot)
  mappings = snapshot["mappings"]
  assert mappings["forward"]["requires_plugin"] == "inject_training_flag"
  assert mappings["parameters"]["requires_plugin"] == "torch_parameters_to_nnx"


def test_flax_nnx_apply_wiring_skip():
  """Verifies the behavior of Flax NNX apply wiring skip."""
  adapter = FlaxNNXAdapter()
  snapshot = {"mappings": {"forward": {"api": "already_set"}}}
  adapter.apply_wiring(snapshot)
  assert snapshot["mappings"]["forward"]["api"] == "already_set"
  assert "requires_plugin" not in snapshot["mappings"]["forward"]


def test_flax_nnx_tiered_examples():
  """Verifies the behavior of Flax NNX tiered examples."""
  adapter = FlaxNNXAdapter()
  examples = adapter.get_tiered_examples()
  assert "tier2_neural" in examples
  assert "class Net(nnx.Module):" in examples["tier2_neural"]


def test_flax_nnx_doc_url():
  """Verifies the behavior of Flax NNX documentation URL."""
  adapter = FlaxNNXAdapter()
  url = adapter.get_doc_url("flax.nnx.relu")
  assert "search.html?q=flax.nnx.relu" in url


def test_flax_nnx_convert_logic(monkeypatch):
  """Test flax nnx convert and logging coverage."""
  import sys
  from unittest.mock import MagicMock
  import logging

  # test logging warning
  adapter = FlaxNNXAdapter()

  with monkeypatch.context() as m:
    m.setitem(sys.modules, "flax.nnx", None)
    import ml_switcheroo.frameworks.flax_nnx as fn_fw

    m.setattr(fn_fw, "flax_nnx", None)

    # intercept load_snapshot_for_adapter returning None
    m.setattr(fn_fw, "load_snapshot_for_adapter", lambda x: None)

    # Test warning
    with m.context() as m2:
      m2.setattr(logging, "warning", MagicMock())
      # adapter2 = fn_fw.FlaxNNXAdapter()
      fn_fw.FlaxNNXAdapter()
      logging.warning.assert_called()

      # also cover live branch
      m2.setattr(fn_fw, "flax_nnx", MagicMock())
      # adapter3 = fn_fw.FlaxNNXAdapter()
      fn_fw.FlaxNNXAdapter()

  # test convert
  import ml_switcheroo.frameworks.flax_nnx as fn_fw

  adapter = fn_fw.FlaxNNXAdapter()
  assert adapter.convert({"a": 1}) == {"a": 1}

  class FakeJNP:
    def array(self, x):
      if x == [1, 2]:
        return "jnp_array"
      raise Exception("fail")

  mock_jax_numpy = FakeJNP()
  mock_jax = MagicMock()
  mock_jax.numpy = mock_jax_numpy
  monkeypatch.setitem(sys.modules, "jax", mock_jax)
  monkeypatch.setitem(sys.modules, "jax.numpy", mock_jax_numpy)
  assert adapter.convert([1, 2]) == "jnp_array"
  assert adapter.convert([3, 4]) == [3, 4]
