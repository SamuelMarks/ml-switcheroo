"""Test suite for the Jax module."""

from ml_switcheroo.enums import RuntimeTensorOrValue

from unittest.mock import patch

import pytest
from ml_switcheroo_ir.schema.ghost import SemanticTier

from ml_switcheroo.enums import SemanticTier  # noqa: F811
from ml_switcheroo.frameworks.base import InitMode
from ml_switcheroo.frameworks.jax import JaxCoreAdapter


def test_jax_adapter_init() -> None:
  """Verifies the behavior of JAX adapter initialization."""
  adapter = JaxCoreAdapter()
  assert adapter.display_name == "JAX (no framework)"
  assert adapter.inherits_from is None
  assert adapter.ui_priority == 10


def test_jax_import_alias() -> None:
  """Verifies the behavior of JAX import alias."""
  adapter = JaxCoreAdapter()
  assert adapter.import_alias == ("jax.numpy", "jnp")


def test_jax_import_namespaces() -> None:
  """Verifies the behavior of JAX import namespaces."""
  adapter = JaxCoreAdapter()
  ns = adapter.import_namespaces
  assert "jax.numpy" in ns
  assert ns["jax.numpy"].recommended_alias == "jnp"
  assert "optax" in ns


def test_jax_test_config() -> None:
  """Verify jax test config behavior."""
  adapter = JaxCoreAdapter()
  config = adapter.test_config
  assert "import jax.numpy as jnp" in config["import"]


def test_jax_harness_imports() -> None:
  """Verifies the behavior of JAX harness imports."""
  adapter = JaxCoreAdapter()
  assert "import jax" in adapter.harness_imports
  assert "import jax.random" in adapter.harness_imports


def test_jax_harness_init_code() -> None:
  """Verifies the behavior of JAX harness initialization code."""
  adapter = JaxCoreAdapter()
  code: str = adapter.get_harness_init_code()
  assert "def _make_jax_key(seed):" in code
  assert "jax.random.PRNGKey(seed)" in code


def test_jax_declared_magic_args() -> None:
  """Verifies the behavior of JAX declared magic arguments."""
  adapter = JaxCoreAdapter()
  assert adapter.declared_magic_args == ["key"]


def test_jax_structural_traits() -> None:
  """Verifies the behavior of JAX structural traits."""
  adapter = JaxCoreAdapter()
  traits = adapter.structural_traits
  assert traits.module_base is None
  assert traits.forward_method == "__call__"
  assert not traits.requires_super_init


def test_jax_rng_seed_methods() -> None:
  """Verifies the behavior of JAX rng seed methods."""
  adapter = JaxCoreAdapter()
  assert adapter.rng_seed_methods == []


def test_jax_definitions(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verifies the behavior of JAX definitions.

  Args:
      monkeypatch: Pytest monkeypatch fixture for modifying attributes or environment.
  """
  adapter = JaxCoreAdapter()
  defs = adapter.definitions
  assert isinstance(defs, dict)


def test_jax_convert_no_array() -> None:
  """Verifies the behavior of JAX convert no array."""
  adapter = JaxCoreAdapter()
  assert adapter.convert("test") == "test"


def test_jax_convert_list() -> None:
  """Verifies the behavior of JAX convert list."""
  adapter = JaxCoreAdapter()
  res = adapter.convert([1, 2, 3])
  assert res is not None


def test_jax_apply_wiring() -> None:
  """Verifies the behavior of JAX apply wiring."""
  adapter = JaxCoreAdapter()
  snapshot: dict[str, RuntimeTensorOrValue] = {}
  getattr({"v": getattr({"v": adapter}, "__getitem__")("v").apply_wiring}, "__getitem__")("v")(
    getattr({"v": snapshot}, "__getitem__")("v")
  )
  assert "mappings" in snapshot
  assert "templates" in snapshot


def test_jax_tiered_examples() -> None:
  """Verifies the behavior of JAX tiered examples."""
  adapter = JaxCoreAdapter()
  examples: dict[str, str] = adapter.get_tiered_examples()
  assert "tier1_math" in examples
  assert "tier2_neural" in examples
  assert "tier4_qwen3-vl" in examples
  assert "tier3_extras" in examples


def test_jax_doc_url() -> None:
  """Verifies the behavior of JAX documentation URL."""
  adapter = JaxCoreAdapter()
  url: str | None = adapter.get_doc_url("jax.numpy.abs")
  assert url == "https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.abs.html"


def test_jax_init_live_mode(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verifies the behavior of JAX initialization live mode.

  Args:
      monkeypatch: Pytest monkeypatch fixture for modifying attributes or environment.
  """
  monkeypatch.setattr("ml_switcheroo.frameworks.jax.jax", True)
  adapter = JaxCoreAdapter()
  assert adapter._mode == InitMode.LIVE


def test_jax_activations_coverage() -> None:
  """Verify jax activations coverage behavior."""
  adapter = JaxCoreAdapter()
  res = adapter._collect_live(SemanticTier.ACTIVATION)
  assert isinstance(res, list)


def test_jax_import_exception() -> None:
  """Test jax module level import failure fallback."""
  import importlib
  import sys
  from unittest.mock import MagicMock

  with patch.dict(sys.modules, {"jax": None, "jax.numpy": None}):
    import ml_switcheroo.frameworks.jax as mod

    importlib.reload(mod)
    assert getattr(mod, "jax") is None
    assert getattr(mod, "jnp") is None

  mock_jax = MagicMock()
  mock_jnp = MagicMock()
  with patch.dict(sys.modules, {"jax": mock_jax, "jax.numpy": mock_jnp}):
    importlib.reload(mod)
    assert getattr(mod, "jax") is not None
    assert getattr(mod, "jnp") is not None

  importlib.reload(mod)


def test_jax_convert_array_exception() -> None:
  """Test jnp.array exception handling during convert."""
  adapter = JaxCoreAdapter()
  import sys
  from unittest.mock import MagicMock

  mock_jax = MagicMock()
  mock_jnp = MagicMock()
  mock_jax.numpy = mock_jnp
  mock_jnp.array.side_effect = ValueError("bad array")
  with patch.dict(sys.modules, {"jax": mock_jax, "jax.numpy": mock_jnp}):
    assert adapter.convert([1, 2, 3]) == [1, 2, 3]
    # Non-array input hits line 257
    assert adapter.convert(42) == 42


def test_jax_adapter_ghost_mode() -> None:
  """Verify jax adapter ghost mode behavior."""
  with patch("ml_switcheroo.frameworks.jax.jax", None):
    with patch(
      "ml_switcheroo.frameworks.jax.load_snapshot_for_adapter",
      return_value={"categories": {"loss": [{"name": "foo", "kind": "function", "api_path": "foo"}]}},
    ):
      adapter = JaxCoreAdapter()
      assert adapter._mode.value == "ghost"

      ghosts = adapter._collect_ghost(SemanticTier.LOSS)
      assert len(ghosts) == 1
      assert ghosts[0].name == "foo"


def test_jax_adapter_ghost_mode_no_snapshot() -> None:
  """Verify jax adapter ghost mode no snapshot behavior."""
  with patch("ml_switcheroo.frameworks.jax.jax", None):
    with patch("ml_switcheroo.frameworks.jax.load_snapshot_for_adapter", return_value={}):
      adapter = JaxCoreAdapter()
      assert adapter._mode.value == "ghost"
      ghosts = adapter._collect_ghost(SemanticTier.LOSS)
      assert len(ghosts) == 0


def test_jax_adapter_plugin_traits() -> None:
  """Verify jax adapter plugin traits behavior."""
  adapter = JaxCoreAdapter()
  traits = adapter.plugin_traits
  assert traits.has_numpy_compatible_arrays


def test_jax_adapter_collect_live() -> None:
  """Verify jax adapter collect live behavior."""
  adapter = JaxCoreAdapter()
  with patch.object(JaxCoreAdapter, "_scan_optax_losses", return_value=["loss1"], create=True):
    assert getattr({"v": "loss1"}, "__getitem__")("v") in getattr(
      {
        "v": getattr({"v": getattr({"v": adapter}, "__getitem__")("v")._collect_live}, "__getitem__")("v")(
          getattr({"v": getattr({"v": SemanticTier}, "__getitem__")("v").LOSS}, "__getitem__")("v")
        )
      },
      "__getitem__",
    )("v")
  with patch.object(JaxCoreAdapter, "_scan_optax_optimizers", return_value=["opt1"], create=True):
    assert getattr({"v": "opt1"}, "__getitem__")("v") in getattr(
      {
        "v": getattr({"v": getattr({"v": adapter}, "__getitem__")("v")._collect_live}, "__getitem__")("v")(
          getattr({"v": getattr({"v": SemanticTier}, "__getitem__")("v").OPTIMIZER}, "__getitem__")("v")
        )
      },
      "__getitem__",
    )("v")
  assert adapter._collect_live(SemanticTier.ARRAY_API) == []


def test_jax_adapter_convert_exception() -> None:
  """Verify jax adapter convert exception behavior."""
  adapter = JaxCoreAdapter()
  with patch("builtins.__import__", side_effect=ImportError):
    assert adapter.convert([1, 2]) == [1, 2]
