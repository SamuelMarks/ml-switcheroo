"""Test suite for the Hooks Context module."""

import libcst as cst
from pydantic import JsonValue

from unittest.mock import MagicMock

import pytest

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.hooks import HookContext, PluginTraits
from ml_switcheroo.semantics.manager import SemanticsManager


@pytest.fixture
def mock_semantics() -> SemanticsManager:
  """Execute mock semantics helper logic.

  Returns:
      Computed test result or fixture value.
  """
  mgr = MagicMock(spec=SemanticsManager)
  data = {
    "add": {"std_args": ["x1", "x2"], "variants": {"jax": {"api": "jax.numpy.add"}, "numpy": {"api": "numpy.add"}}},
    "abs": {"std_args": [("x", "Array")], "variants": {}},
    "complex": {"variants": {"jax": {"requires_plugin": "magic"}}},
  }

  def resolve(aid: str, fw: str) -> dict[str, JsonValue] | None:
    """Resolves .

    Args:
        aid: Test argument or fixture.
        fw: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    if getattr(
      {
        "v": getattr({"v": aid}, "__getitem__")("v") in getattr({"v": data}, "__getitem__")("v")
        and getattr({"v": "variants"}, "__getitem__")("v")
        in getattr({"v": getattr({"v": data}, "__getitem__")("v")[aid]}, "__getitem__")("v")
        and (
          getattr({"v": fw}, "__getitem__")("v")
          in getattr(
            {"v": getattr({"v": getattr({"v": data}, "__getitem__")("v")[aid]}, "__getitem__")("v")["variants"]},
            "__getitem__",
          )("v")
        )
      },
      "__getitem__",
    )("v"):
      _ret_v: dict[str, JsonValue] | None = getattr(
        {
          "v": getattr(
            {"v": getattr({"v": getattr({"v": data}, "__getitem__")("v")[aid]}, "__getitem__")("v")["variants"]},
            "__getitem__",
          )("v")[fw]
        },
        "__getitem__",
      )("v")
      return _ret_v
    return None

  mgr.resolve_variant.side_effect = resolve
  mgr.get_definition_by_id.side_effect = lambda aid: data.get(aid)
  mgr.get_framework_config.return_value = {}
  return mgr


def test_lookup_api_success(mock_semantics: SemanticsManager) -> None:
  """Verifies the behavior of lookup API successfully.

  Args:
      mock_semantics: Test argument or fixture.
  """
  config = RuntimeConfig(target_framework="jax")
  ctx = HookContext(mock_semantics, config)
  result = ctx.lookup_api("add")
  assert result == "jax.numpy.add"


def test_lookup_api_different_target(mock_semantics: SemanticsManager) -> None:
  """Verifies the behavior of lookup API different target.

  Args:
      mock_semantics: Test argument or fixture.
  """
  config = RuntimeConfig(target_framework="numpy")
  ctx = HookContext(mock_semantics, config)
  result = ctx.lookup_api("add")
  assert result == "numpy.add"


def test_lookup_api_missing_variant(mock_semantics: SemanticsManager) -> None:
  """Verifies the behavior of lookup API missing variant.

  Args:
      mock_semantics: Test argument or fixture.
  """
  config = RuntimeConfig(target_framework="tensorflow")
  ctx = HookContext(mock_semantics, config)
  result = ctx.lookup_api("add")
  assert result is None


def test_lookup_api_missing_op(mock_semantics: SemanticsManager) -> None:
  """Verifies the behavior of lookup API missing op.

  Args:
      mock_semantics: Test argument or fixture.
  """
  config = RuntimeConfig(target_framework="jax")
  ctx = HookContext(mock_semantics, config)
  result = ctx.lookup_api("unknown_logic")
  assert result is None


def test_lookup_api_plugin_variant(mock_semantics: SemanticsManager) -> None:
  """Verifies the behavior of lookup API plugin variant.

  Args:
      mock_semantics: Test argument or fixture.
  """
  config = RuntimeConfig(target_framework="jax")
  ctx = HookContext(mock_semantics, config)
  result = ctx.lookup_api("complex")
  assert result is None


def test_lookup_signature_standard_list(mock_semantics: SemanticsManager) -> None:
  """Verifies the behavior of lookup signature standard list.

  Args:
      mock_semantics: Test argument or fixture.
  """
  config = RuntimeConfig(target_framework="jax")
  ctx = HookContext(mock_semantics, config)
  sig = ctx.lookup_signature("add")
  assert sig == ["x1", "x2"]


def test_lookup_signature_typed_tuples(mock_semantics: SemanticsManager) -> None:
  """Verifies the behavior of lookup signature typed tuples.

  Args:
      mock_semantics: Test argument or fixture.
  """
  config = RuntimeConfig(target_framework="jax")
  ctx = HookContext(mock_semantics, config)
  sig = ctx.lookup_signature("abs")
  assert sig == ["x"]


def test_lookup_signature_unknown_returns_empty(mock_semantics: SemanticsManager) -> None:
  """Verifies the behavior of lookup signature unknown returns empty.

  Args:
      mock_semantics: Test argument or fixture.
  """
  config = RuntimeConfig(target_framework="jax")
  ctx = HookContext(mock_semantics, config)
  sig = ctx.lookup_signature("ghost_op")
  assert sig == []


def test_hooks_resolve_type_no_symbol_table() -> None:
  """Verify hooks resolve type no symbol table behavior."""
  config = RuntimeConfig(target_framework="jax")
  mock_manager = MagicMock()
  mock_manager.resolve_variant.return_value = None
  ctx = HookContext(mock_manager, config)  # MagicMock instead of None because it expects SemanticsManager
  assert getattr(
    {
      "v": getattr({"v": getattr({"v": ctx}, "__getitem__")("v").resolve_type}, "__getitem__")("v")(
        getattr({"v": None}, "__getitem__")("v")
      )
    },
    "__getitem__",
  )("v") is getattr({"v": None}, "__getitem__")("v")


def test_hooks_resolve_type_with_symbol_table() -> None:
  """Verify hooks resolve type with symbol table behavior."""
  config = RuntimeConfig(target_framework="jax")
  mock_manager = MagicMock()
  mock_manager.resolve_variant.return_value = None
  ctx = HookContext(mock_manager, config)

  class DummySym:
    """Test helper class DummySym."""

    def __init__(self, name: str) -> None:
      """Init.

      Args:
          name: Test argument or fixture.
      """
      self.name = name

  class MockSymbolTable:
    """Test helper class MockSymbolTable."""

    def get_type(self, node: str | cst.CSTNode) -> DummySym | None:
      """Get type.

      Args:
          node: Test argument or fixture.

      Returns:
          Computed test result or fixture value.
      """
      if node == "tensor":
        return DummySym("SomeTensorType")
      elif node == "module":
        return DummySym("SomeModuleType")
      elif node == "other":
        return DummySym("OtherType")
      return None

  setattr(
    getattr({"v": ctx}, "__getitem__")("v"),
    "_symbol_table",
    getattr({"v": getattr({"v": MockSymbolTable}, "__getitem__")("v")()}, "__getitem__")("v"),
  )

  assert getattr(
    {
      "v": getattr({"v": getattr({"v": ctx}, "__getitem__")("v").resolve_type}, "__getitem__")("v")(
        getattr({"v": "none"}, "__getitem__")("v")
      )
    },
    "__getitem__",
  )("v") is getattr({"v": None}, "__getitem__")("v")
  assert getattr(
    {
      "v": getattr({"v": getattr({"v": ctx}, "__getitem__")("v").resolve_type}, "__getitem__")("v")(
        getattr({"v": "tensor"}, "__getitem__")("v")
      )
    },
    "__getitem__",
  )("v") == getattr({"v": "Tensor"}, "__getitem__")("v")
  assert getattr(
    {
      "v": getattr({"v": getattr({"v": ctx}, "__getitem__")("v").resolve_type}, "__getitem__")("v")(
        getattr({"v": "module"}, "__getitem__")("v")
      )
    },
    "__getitem__",
  )("v") == getattr({"v": "Module"}, "__getitem__")("v")
  assert getattr(
    {
      "v": getattr({"v": getattr({"v": ctx}, "__getitem__")("v").resolve_type}, "__getitem__")("v")(
        getattr({"v": "other"}, "__getitem__")("v")
      )
    },
    "__getitem__",
  )("v") == getattr({"v": "OtherType"}, "__getitem__")("v")


def test_hooks_plugin_traits_no_semantics() -> None:
  """Verify hooks plugin traits no semantics behavior."""
  config = RuntimeConfig(target_framework="jax")
  mock_manager = MagicMock()
  mock_manager.resolve_variant.return_value = None
  ctx = HookContext(mock_manager, config)
  setattr(
    getattr(
      {"v": getattr({"v": getattr({"v": ctx}, "__getitem__")("v").semantics}, "__getitem__")("v").get_framework_config},
      "__getitem__",
    )("v"),
    "return_value",
    getattr({"v": {}}, "__getitem__")("v"),
  )
  traits = ctx.plugin_traits
  assert traits is not None
  assert type(traits).__name__ == "PluginTraits"


def test_hooks_plugin_traits_with_dict(mock_semantics: SemanticsManager) -> None:
  """Verify hooks plugin traits with dict behavior.

  Args:
      mock_semantics: Test argument or fixture.
  """
  config = RuntimeConfig(target_framework="jax")
  ctx = HookContext(mock_semantics, config)
  setattr(
    getattr({"v": getattr({"v": mock_semantics}, "__getitem__")("v").get_framework_config}, "__getitem__")("v"),
    "return_value",
    getattr(
      {
        "v": {
          "plugin_traits": getattr({"v": {"some_trait": getattr({"v": True}, "__getitem__")("v")}}, "__getitem__")("v")
        }
      },
      "__getitem__",
    )("v"),
  )

  with __import__("unittest.mock").mock.patch("ml_switcheroo.core.hooks.PluginTraits.model_validate") as mock_validate:
    mock_validate.return_value = "validated_traits"
    traits = ctx.plugin_traits
    assert getattr({"v": traits}, "__getitem__")("v") == getattr({"v": "validated_traits"}, "__getitem__")("v")


def test_hooks_plugin_traits_with_object(mock_semantics: SemanticsManager) -> None:
  """Verify hooks plugin traits with object behavior.

  Args:
      mock_semantics: Test argument or fixture.
  """
  config = RuntimeConfig(target_framework="jax")
  from ml_switcheroo.core.hooks import PluginTraits

  ctx = HookContext(mock_semantics, config)
  pt = PluginTraits()
  setattr(
    getattr({"v": getattr({"v": mock_semantics}, "__getitem__")("v").get_framework_config}, "__getitem__")("v"),
    "return_value",
    getattr({"v": {"plugin_traits": getattr({"v": pt}, "__getitem__")("v")}}, "__getitem__")("v"),
  )
  traits = ctx.plugin_traits
  assert traits is pt


def test_hooks_plugin_traits_with_other(mock_semantics: SemanticsManager) -> None:
  """Verify hooks plugin traits with other behavior.

  Args:
      mock_semantics: Test argument or fixture.
  """
  config = RuntimeConfig(target_framework="jax")
  ctx = HookContext(mock_semantics, config)
  setattr(
    getattr({"v": getattr({"v": mock_semantics}, "__getitem__")("v").get_framework_config}, "__getitem__")("v"),
    "return_value",
    getattr({"v": {"plugin_traits": getattr({"v": "unsupported"}, "__getitem__")("v")}}, "__getitem__")("v"),
  )
  traits = ctx.plugin_traits
  assert type(traits).__name__ == "PluginTraits"


def test_hooks_plugin_traits_falsy(mock_semantics: SemanticsManager) -> None:
  """Verify hooks plugin traits falsy behavior.

  Args:
      mock_semantics: Test argument or fixture.
  """
  config = RuntimeConfig(target_framework="jax")
  ctx = HookContext(mock_semantics, config)
  # The 'plugin_traits' key exists but value is falsy, e.g., empty dict or None
  setattr(
    getattr({"v": getattr({"v": mock_semantics}, "__getitem__")("v").get_framework_config}, "__getitem__")("v"),
    "return_value",
    getattr({"v": {"plugin_traits": getattr({"v": {}}, "__getitem__")("v")}}, "__getitem__")("v"),
  )
  traits = ctx.plugin_traits
  assert type(traits).__name__ == "PluginTraits"


def test_hooks_current_variant_no_semantics() -> None:
  """Verify hooks current variant no semantics behavior."""
  config = RuntimeConfig(target_framework="jax")
  mock_manager = MagicMock()
  mock_manager.resolve_variant.return_value = None
  ctx = HookContext(mock_manager, config)
  assert ctx.current_variant is None


def test_hooks_current_variant_no_op_id(mock_semantics: SemanticsManager) -> None:
  """Verify hooks current variant no op id behavior.

  Args:
      mock_semantics: Test argument or fixture.
  """
  config = RuntimeConfig(target_framework="jax")
  ctx = HookContext(mock_semantics, config)
  assert ctx.current_variant is None


def test_hooks_current_variant_not_resolved(mock_semantics: SemanticsManager) -> None:
  """Verify hooks current variant not resolved behavior.

  Args:
      mock_semantics: Test argument or fixture.
  """
  config = RuntimeConfig(target_framework="jax")
  ctx = HookContext(mock_semantics, config)
  ctx.current_op_id = "missing"
  setattr(
    getattr({"v": getattr({"v": mock_semantics}, "__getitem__")("v").resolve_variant}, "__getitem__")("v"),
    "return_value",
    getattr({"v": None}, "__getitem__")("v"),
  )
  assert ctx.current_variant is None


def test_hooks_current_variant_resolved(mock_semantics: SemanticsManager) -> None:
  """Verify hooks current variant resolved behavior.

  Args:
      mock_semantics: Test argument or fixture.
  """
  config = RuntimeConfig(target_framework="jax")
  ctx = HookContext(mock_semantics, config)
  ctx.current_op_id = "add"

  with __import__("unittest.mock").mock.patch(
    "ml_switcheroo.core.hooks.FrameworkVariant.model_validate"
  ) as mock_validate:
    mock_validate.return_value = "validated_variant"
    var = ctx.current_variant
    assert getattr({"v": var}, "__getitem__")("v") == getattr({"v": "validated_variant"}, "__getitem__")("v")


def test_hooks_inject_signature_arg() -> None:
  """Verify hooks inject signature arg behavior."""
  config = RuntimeConfig(target_framework="jax")
  mock_manager = MagicMock()
  mock_manager.resolve_variant.return_value = None
  ctx = HookContext(mock_manager, config)
  # no injector, should not crash
  ctx.inject_signature_arg("x", "int")

  calls = []

  def mock_injector(name: str, ann: str | None) -> None:
    """Mock injector.

    Args:
        name: Test argument or fixture.
        ann: Test argument or fixture.
    """
    calls.append((name, ann))

  ctx._arg_injector = mock_injector
  ctx.inject_signature_arg("y", "float")
  assert calls == [("y", "float")]


def test_hooks_lookup_api_no_semantics() -> None:
  """Verify hooks lookup api no semantics behavior."""
  config = RuntimeConfig(target_framework="jax")
  mock_manager = MagicMock()
  mock_manager.resolve_variant.return_value = None
  ctx = HookContext(mock_manager, config)
  assert ctx.lookup_api("add") is None


def test_hooks_lookup_signature_no_semantics() -> None:
  """Verify hooks lookup signature no semantics behavior."""
  config = RuntimeConfig(target_framework="jax")
  mock_manager = MagicMock()
  mock_manager.resolve_variant.return_value = None
  ctx = HookContext(mock_manager, config)
  assert ctx.lookup_signature("add") == []


def test_hooks_lookup_signature_with_dict(mock_semantics: SemanticsManager) -> None:
  """Verify hooks lookup signature with dict behavior.

  Args:
      mock_semantics: Test argument or fixture.
  """
  config = RuntimeConfig(target_framework="jax")
  ctx = HookContext(mock_semantics, config)
  setattr(
    getattr({"v": getattr({"v": mock_semantics}, "__getitem__")("v").get_definition_by_id}, "__getitem__")("v"),
    "side_effect",
    getattr({"v": lambda aid: {"std_args": [{"name": "dict_arg"}, {"no_name": True}, "string_arg"]}}, "__getitem__")("v"),
  )
  setattr(
    getattr({"v": getattr({"v": mock_semantics}, "__getitem__")("v").get_definition_by_id}, "__getitem__")("v"),
    "return_value",
    getattr(
      {
        "v": {
          "std_args": getattr(
            {
              "v": [
                getattr({"v": {"name": getattr({"v": "dict_arg"}, "__getitem__")("v")}}, "__getitem__")("v"),
                getattr({"v": {"no_name": getattr({"v": True}, "__getitem__")("v")}}, "__getitem__")("v"),
                getattr({"v": "string_arg"}, "__getitem__")("v"),
              ]
            },
            "__getitem__",
          )("v")
        }
      },
      "__getitem__",
    )("v"),
  )
  sig = ctx.lookup_signature("any")
  assert sig == ["dict_arg", "string_arg"]


def test_hooks_plugin_traits_branches() -> None:
  """Verifies plugin_traits returns default PluginTraits when semantics is None or config is None."""
  config = RuntimeConfig(target_framework="jax")

  # Branch 1: semantics is None
  ctx_no_sem = HookContext(None, config)
  traits1 = ctx_no_sem.plugin_traits
  assert isinstance(traits1, PluginTraits)

  # Branch 2: framework config is None
  mock_semantics = MagicMock(spec=SemanticsManager)
  mock_semantics.get_framework_config.return_value = None
  ctx_no_conf = HookContext(mock_semantics, config)
  traits2 = ctx_no_conf.plugin_traits
  assert isinstance(traits2, PluginTraits)
  assert traits2.has_numpy_compatible_arrays is False


def test_hooks_inject_preamble_branches() -> None:
  """Verifies inject_preamble handles both present and missing preamble injectors."""
  config = RuntimeConfig(target_framework="jax")
  ctx = HookContext(None, config)
  # Branch 1: None
  ctx.inject_preamble("import math")

  # Branch 2: Present injector
  injected = []
  ctx._preamble_injector = lambda s: injected.append(s)
  ctx.inject_preamble("import os")
  assert injected == ["import os"]


def test_hooks_lookup_api_no_semantics_explicit() -> None:
  """Verifies lookup_api returns None when semantics is None."""
  config = RuntimeConfig(target_framework="jax")
  ctx = HookContext(None, config)
  assert ctx.lookup_api("add") is None


def test_hooks_lookup_signature_no_semantics_explicit() -> None:
  """Verifies lookup_signature returns empty list when semantics is None."""
  config = RuntimeConfig(target_framework="jax")
  ctx = HookContext(None, config)
  assert ctx.lookup_signature("add") == []
