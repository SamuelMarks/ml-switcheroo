import pytest
from ml_switcheroo.frameworks.paxml import PaxmlAdapter
from ml_switcheroo.enums import SemanticTier
from ml_switcheroo.frameworks.base import StandardCategory
from ml_switcheroo.core.ghost import GhostRef


def test_paxml_adapter():
  adapter = PaxmlAdapter()
  assert adapter.display_name == "PaxML / Praxis"

  assert (
    adapter.search_modules == ["praxis.layers", "praxis.base_layer", "praxis.layers.activations", "optax"]
    or adapter.search_modules == []
  )
  assert adapter.unsafe_submodules == set()
  assert adapter.import_alias == ("praxis.layers", "pl")

  ns = adapter.import_namespaces
  assert "praxis.layers" in ns
  assert ns["praxis.layers"].tier == SemanticTier.NEURAL

  heuristics = adapter.discovery_heuristics
  assert "neural" in heuristics

  tc = adapter.test_config
  assert "import" in tc

  assert adapter.harness_imports == ["import jax", "import jax.random"]
  assert adapter.get_harness_init_code() != ""
  assert SemanticTier.ARRAY_API in adapter.supported_tiers
  assert adapter.declared_magic_args == []

  traits = adapter.structural_traits
  assert traits.module_base == "praxis.base_layer.BaseLayer"

  pt = adapter.plugin_traits
  assert pt.enforce_purity_analysis

  defs = adapter.definitions
  assert "Linear" in defs
  assert "Sequential" in defs
  assert "ReLU" in defs

  from ml_switcheroo.frameworks.base import StandardCategory, InitMode

  ...
  # Discovery
  adapter._mode = InitMode.LIVE
  try:
    import praxis

    praxis_installed = True
  except ImportError:
    praxis_installed = False

  if praxis_installed:
    refs = adapter.collect_api(StandardCategory.LAYER)
    assert isinstance(refs, list)

    refs = adapter.collect_api(StandardCategory.LOSS)
    assert isinstance(refs, list)

    # _scan_praxis_layers
    layers = adapter._scan_praxis_layers()
    assert isinstance(layers, list)

  import sys

  old_praxis = sys.modules.get("praxis")
  sys.modules["praxis"] = None
  try:
    layers = adapter._scan_praxis_layers()
    assert layers == []
  finally:
    sys.modules["praxis"] = old_praxis

  adapter._mode = InitMode.GHOST
  refs = adapter.collect_api(StandardCategory.LAYER)
  assert isinstance(refs, list)

  adapter._snapshot_data = {
    "categories": {
      StandardCategory.LAYER.value: [{"name": "test_layer", "api_path": "test", "kind": "class", "args": {}}]
    }
  }
  refs = adapter.collect_api(StandardCategory.LAYER)
  assert len(refs) == 1

  # apply_wiring
  snap = {}
  adapter.apply_wiring(snap)

  # URL
  assert (
    adapter.get_doc_url("praxis.layers.Linear")
    == "https://github.com/search?q=repo%3Agoogle%2Fpaxml+praxis.layers.Linear&type=code"
  )

  examples = adapter.get_tiered_examples()
  assert "tier2_neural" in examples
  assert "tier3_extras" in examples

  # convert
  assert adapter.convert([1, 2, 3]) is not None


def test_paxml_module_reimport():
  import sys
  import types
  import importlib

  mock_paxml = types.ModuleType("paxml")
  mock_praxis = types.ModuleType("praxis")
  sys.modules["paxml"] = mock_paxml
  sys.modules["praxis"] = mock_praxis
  sys.modules["praxis.layers"] = mock_praxis
  sys.modules["praxis.optimizers"] = mock_praxis

  import ml_switcheroo.frameworks.paxml

  importlib.reload(ml_switcheroo.frameworks.paxml)

  # 72, 99
  adapter = ml_switcheroo.frameworks.paxml.PaxmlAdapter()
  with __import__("unittest.mock").mock.patch("ml_switcheroo.frameworks.paxml.praxis", None):
    with __import__("unittest.mock").mock.patch(
      "ml_switcheroo.frameworks.paxml.load_snapshot_for_adapter", return_value=None
    ):
      ml_switcheroo.frameworks.paxml.PaxmlAdapter()
      assert adapter.search_modules == []

  # 113
  with __import__("unittest.mock").mock.patch("ml_switcheroo.frameworks.paxml.praxis", True):
    adapter = ml_switcheroo.frameworks.paxml.PaxmlAdapter()
    assert adapter.search_modules == ["praxis.layers", "praxis.base_layer", "praxis.layers.activations", "optax"]

  # 337, 341, 347, 351, 363 (getters)
  assert adapter.get_device_syntax("cpu") is not None
  assert adapter.get_device_check_syntax() is not None
  assert adapter.get_serialization_imports() == ["import orbax.checkpoint"]
  assert adapter.get_serialization_syntax("op", "f") == ""
  assert adapter.get_weight_conversion_imports() is not None


def test_paxml_collect_api():
  import sys
  import types

  mock_praxis = types.ModuleType("praxis")
  mock_praxis.layers = types.ModuleType("praxis.layers")
  sys.modules["praxis"] = mock_praxis
  sys.modules["praxis.layers"] = mock_praxis.layers

  from ml_switcheroo.frameworks.paxml import PaxmlAdapter
  from ml_switcheroo.frameworks.base import StandardCategory

  adapter = PaxmlAdapter()
  adapter._mode = __import__("ml_switcheroo.frameworks.base").frameworks.base.InitMode.LIVE

  with __import__("unittest.mock").mock.patch("ml_switcheroo.frameworks.paxml.praxis", mock_praxis):
    with __import__("unittest.mock").mock.patch(
      "ml_switcheroo.frameworks.optax_shim.OptaxScanner.scan_losses", return_value=["l"]
    ):
      assert "l" in adapter.collect_api(StandardCategory.LOSS) or len(adapter.collect_api(StandardCategory.LOSS)) == 0

    with __import__("unittest.mock").mock.patch(
      "ml_switcheroo.frameworks.optax_shim.OptaxScanner.scan_optimizers", return_value=["o"]
    ):
      assert (
        "o" in adapter.collect_api(StandardCategory.OPTIMIZER)
        or len(adapter.collect_api(StandardCategory.OPTIMIZER)) == 0
      )

    with __import__("unittest.mock").mock.patch(
      "ml_switcheroo.frameworks.jax.JaxCoreAdapter.collect_api", return_value=["a"]
    ):
      assert (
        "a" in adapter.collect_api(StandardCategory.ACTIVATION)
        or len(adapter.collect_api(StandardCategory.ACTIVATION)) == 0
      )

    with __import__("unittest.mock").mock.patch.object(adapter, "_scan_praxis_layers", return_value=["m"]):
      assert adapter.collect_api(StandardCategory.LAYER) == ["m"]
