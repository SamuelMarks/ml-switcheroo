"""Docstring."""

from ml_switcheroo.semantics.manager import SemanticsManager


def test_manager_init():
  """Docstring."""
  sm = SemanticsManager()
  assert sm is not None


def test_manager_coverage():
  """Docstring."""
  from ml_switcheroo.semantics.manager import SemanticsManager

  sm = SemanticsManager()

  assert sm.get_test_template("not_real") is None

  assert sm.get_framework_aliases() is not None

  assert sm.get_all_rng_methods() is not None

  assert sm.get_patterns() is not None


def test_manager_load_validation():
  """Docstring."""
  from ml_switcheroo.semantics.manager import SemanticsManager
  from pathlib import Path
  from unittest.mock import patch, mock_open

  sm = SemanticsManager()

  # Path doesn't exist
  with patch("pathlib.Path.exists", return_value=False):
    sm.load_validation_report(Path("dummy.json"))

  # Read err
  with patch("pathlib.Path.exists", return_value=True):
    with patch("builtins.open", side_effect=Exception("Read err")):
      sm.load_validation_report(Path("dummy.json"))

  # Success
  m_open = mock_open(read_data='{"Abs": true}')
  with patch("pathlib.Path.exists", return_value=True):
    with patch("builtins.open", m_open):
      sm.load_validation_report(Path("dummy.json"))
      assert sm._validation_status.get("Abs") is True


def test_manager_update_definition():
  """Docstring."""
  from ml_switcheroo.semantics.manager import SemanticsManager
  from unittest.mock import patch, mock_open

  sm = SemanticsManager()

  # Validation error
  sm.update_definition("Abs", {"operation": "Abs", "std_args": "NOT_A_LIST"})  # Will trigger Pydantic error

  # Valid
  m_open = mock_open()
  with patch("builtins.open", m_open):
    with patch("pathlib.Path.mkdir"):
      sm.update_definition("NewOp", {"variants": {"torch": {"api": "torch.new_op"}}})
      assert "NewOp" in sm.data
      assert "torch.new_op" in sm._reverse_index

  # Write error
  with patch("builtins.open", side_effect=Exception("Write err")):
    with patch("pathlib.Path.mkdir"):
      sm.update_definition("WriteFail", {"variants": {"torch": {"api": "torch.write_fail"}}})
      assert "WriteFail" in sm.data


def test_manager_get_import_map():
  """Docstring."""
  from ml_switcheroo.semantics.manager import SemanticsManager

  sm = SemanticsManager()

  # Mocking internal states to hit coverage
  sm._providers = {
    "jax": {"core": {"root": "jax.numpy", "alias": "jnp"}},
    "flax": {"nn": {"root": "flax.linen", "alias": "nn"}},
  }
  sm._source_registry = {
    "torch": ("torch", "core"),
    "torch.nn": ("torch.nn", "nn"),
    "torch.optim": ("torch.optim", "optim"),  # unmatched
  }

  # Try direct mapping
  res = sm.get_import_map("jax")
  assert "torch" in res
  assert res["torch"][0] == "jax.numpy"

  # Try inheritance mapping
  # Mock _resolve_inheritance to return "flax" for some target
  sm._framework_aliases = {"myfw": ("myfw.mod", "myfw")}  # Doesn't matter

  # Let's override resolve_inheritance
  original_res = sm._resolve_inheritance
  sm._resolve_inheritance = lambda x: "flax" if x == "custom" else None

  res2 = sm.get_import_map("custom")
  assert "torch.nn" in res2
  assert res2["torch.nn"][0] == "flax.linen"

  sm._resolve_inheritance = original_res


def test_manager_resolve_variant():
  """Docstring."""
  from ml_switcheroo.semantics.manager import SemanticsManager

  sm = SemanticsManager()

  # Mock data
  sm.data = {"Abs": {"variants": {"torch": {"api": "torch.abs"}, "numpy": {"api": "np.abs"}}}}

  # Direct match
  assert sm.resolve_variant("Abs", "torch") is not None
  assert sm.resolve_variant("Abs", "torch")["api"] == "torch.abs"

  # Missing
  assert sm.resolve_variant("Unknown", "torch") is None

  # Fallback inheritance match
  sm._resolve_inheritance = lambda x: "numpy" if x == "custom_numpy" else None

  assert sm.resolve_variant("Abs", "custom_numpy") is not None
  assert sm.resolve_variant("Abs", "custom_numpy")["api"] == "np.abs"

  # Fallback missing
  assert sm.resolve_variant("Abs", "unknown_fw") is None


def test_manager_get_definition_missing():
  """Docstring."""
  from ml_switcheroo.semantics.manager import SemanticsManager

  sm = SemanticsManager()
  assert sm.get_definition("unknown.api") is None


def test_manager_get_framework_config():
  """Docstring."""
  from ml_switcheroo.semantics.manager import SemanticsManager

  sm = SemanticsManager()
  sm._providers = {"jax": {"core": {"provider": "val"}}}

  res = sm.get_framework_config("jax")
  assert res is not None

  sm._resolve_inheritance = lambda x: "jax" if x == "custom" else None
  sm.get_framework_config("custom")
  assert res is not None

  assert sm.get_framework_config("unknown") == {}


def test_manager_resolve_inheritance():
  """Docstring."""
  from ml_switcheroo.semantics.manager import SemanticsManager

  sm = SemanticsManager()
  sm.framework_configs = {"custom": {"extends": "base_custom"}}
  assert sm._resolve_inheritance("custom") == "base_custom"

  # testing adapter inheriting
  # We mock get_adapter to return a mock adapter with inherits_from
  from unittest.mock import patch, MagicMock

  mock_ad = MagicMock()
  mock_ad.inherits_from = "base_ad"
  with patch("ml_switcheroo.semantics.manager.get_adapter", return_value=mock_ad):
    assert sm._resolve_inheritance("custom2") == "base_ad"


def test_manager_resolve_variant_limit():
  """Docstring."""
  from ml_switcheroo.semantics.manager import SemanticsManager

  sm = SemanticsManager()
  sm.data = {"Abs": {"variants": {"root": {"api": "root.api"}}}}
  # Create an inheritance cycle
  sm._resolve_inheritance = lambda x: "b" if x == "a" else "a"
  assert sm.resolve_variant("Abs", "a") is None


def test_manager_is_verified():
  """Docstring."""
  from ml_switcheroo.semantics.manager import SemanticsManager

  sm = SemanticsManager()
  sm._validation_status = {"Abs": False, "Relu": True}
  assert sm.is_verified("Abs") is False
  assert sm.is_verified("Relu") is True
  assert sm.is_verified("Unknown") is True


def test_manager_get_definition_by_id():
  """Docstring."""
  from ml_switcheroo.semantics.manager import SemanticsManager

  sm = SemanticsManager()
  sm.data = {"Abs": {"foo": "bar"}}
  assert sm.get_definition_by_id("Abs") == {"foo": "bar"}


def test_manager_get_definition():
  """Docstring."""
  from ml_switcheroo.semantics.manager import SemanticsManager

  sm = SemanticsManager()
  sm.data = {"torch.abs": {"api": "torch.abs"}}
  res = sm.get_definition("torch.abs")
  assert res is not None
  assert res[0] == "Abs"


def test_manager_get_definition_missing_real():
  """Docstring."""
  from ml_switcheroo.semantics.manager import SemanticsManager

  sm = SemanticsManager()
  sm.data = {}
  sm._reverse_index = {}
  assert sm.get_definition("completely.unknown.api") is None
