"""Test suite for the Paxml module."""

import typing
import pytest
from ml_switcheroo.frameworks.paxml import PaxmlAdapter
from ml_switcheroo.frameworks.base import InitMode
from ml_switcheroo_ir.schema.ghost import SemanticTier


def test_paxml_adapter_init() -> None:
  """Verifies the behavior of Paxml adapter initialization."""
  adapter = PaxmlAdapter()
  assert adapter.display_name == "PaxML / Praxis"
  assert adapter.inherits_from == "jax"
  assert adapter.ui_priority == 60
  assert adapter._mode == InitMode.GHOST


def test_paxml_import_alias() -> None:
  """Verifies the behavior of Paxml import alias."""
  adapter = PaxmlAdapter()
  assert adapter.import_alias == ("praxis.layers", "pl")


def test_paxml_import_namespaces() -> None:
  """Verifies the behavior of Paxml import namespaces."""
  adapter = PaxmlAdapter()
  ns: typing.Any = adapter.import_namespaces
  assert "praxis.layers" in ns
  assert "praxis.base_layer" in ns


def test_paxml_test_config() -> None:
  """Verifies the behavior of Paxml test configuration."""
  adapter = PaxmlAdapter()
  config: dict[str, typing.Any] = adapter.test_config
  assert "import praxis.layers as pl" in config["import"]


def test_paxml_harness_imports() -> None:
  """Verifies the behavior of Paxml harness imports."""
  adapter = PaxmlAdapter()
  assert "import jax" in adapter.harness_imports


def test_paxml_harness_init_code() -> None:
  """Verifies the behavior of Paxml harness initialization code."""
  adapter = PaxmlAdapter()
  code: str = adapter.get_harness_init_code()
  assert "def _make_jax_key(seed):" in code


def test_paxml_supported_tiers() -> None:
  """Verifies the behavior of Paxml supported tiers."""
  adapter = PaxmlAdapter()
  assert SemanticTier.ARRAY_API in adapter.supported_tiers
  assert SemanticTier.NEURAL in adapter.supported_tiers


def test_paxml_declared_magic_args() -> None:
  """Verifies the behavior of Paxml declared magic arguments."""
  adapter = PaxmlAdapter()
  assert adapter.declared_magic_args == []


def test_paxml_structural_traits() -> None:
  """Verifies the behavior of Paxml structural traits."""
  adapter = PaxmlAdapter()
  traits: typing.Any = adapter.structural_traits
  assert traits.module_base == "praxis.base_layer.BaseLayer"
  assert traits.init_method_name == "setup"
  assert traits.forward_method == "__call__"
  assert not traits.requires_super_init


def test_paxml_definitions() -> None:
  """Verifies the behavior of Paxml definitions."""
  adapter = PaxmlAdapter()
  defs: typing.Any = adapter.definitions
  assert "Linear" in defs
  assert defs["Linear"].args["in_features"] == "input_dims"
  assert defs["Linear"].args["bias"] == "use_bias"


def test_paxml_rng_seed_methods() -> None:
  """Verifies the behavior of Paxml rng seed methods."""
  adapter = PaxmlAdapter()
  assert adapter.rng_seed_methods == []


def test_paxml_convert() -> None:
  """Verifies the behavior of Paxml convert."""
  adapter = PaxmlAdapter()
  assert adapter.convert("test") == "test"


def test_paxml_apply_wiring() -> None:
  """Verifies the behavior of Paxml apply wiring."""
  adapter = PaxmlAdapter()
  snapshot: dict[str, typing.Any] = {}
  adapter.apply_wiring(snapshot)
  assert "mappings" in snapshot
  assert "templates" in snapshot


def test_paxml_doc_url() -> None:
  """Verifies the behavior of Paxml documentation URL."""
  adapter = PaxmlAdapter()
  url: typing.Optional[str] = adapter.get_doc_url("praxis.layers.Linear")
  assert url is not None
  assert "github.com" in url


def test_paxml_tiered_examples() -> None:
  """Verifies the behavior of Paxml tiered examples."""
  adapter = PaxmlAdapter()
  examples: dict[str, str] = adapter.get_tiered_examples()
  assert "tier1_math" in examples
  assert "tier2_neural" in examples
  assert "tier3_extras" in examples
  assert "tier4_qwen3-vl" in examples


def test_paxml_init_live_mode(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verifies the behavior of Paxml initialization live mode."""
  monkeypatch.setattr("ml_switcheroo.frameworks.paxml.praxis", True)
  adapter = PaxmlAdapter()
  assert adapter._mode == InitMode.LIVE
