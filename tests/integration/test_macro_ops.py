"""Test suite for the Macro Ops module."""

from ml_switcheroo.enums import RuntimeTensorOrValue

from pydantic import JsonValue

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.engine import ASTEngine, ConversionResult
from ml_switcheroo.semantics.manager import SemanticsManager

SOURCE_CODE: str = (
  "\nimport torch\n\ndef activation(x):\n    s = torch.swish(x)\n    m = torch.mish(x)\n    return s, m\n"
)
EXPECTED_JAX: str = "\nimport jax.numpy as jnp\nimport jax.nn as nn\n\ndef activation(x):\n    s = x * nn.sigmoid(x)\n    m = x * jnp.tanh(nn.softplus(x))\n    return s, m\n"


class MacroSemantics(SemanticsManager):
  """Test helper class MacroSemantics."""

  def __init__(self) -> None:
    """Initializes the MacroSemantics instance."""
    self.data = {}
    self._providers = {}
    self._source_registry = {}
    self.framework_configs = {}
    self._reverse_index = {}
    self._key_origins: dict[str, str] = {}
    self._validation_status = {}
    self._known_rng_methods: set[str] = set()
    self.import_data: dict[str, RuntimeTensorOrValue] = {}
    self._inject("Swish", ["x"], source="torch.swish", target_macro="{x} * jax.nn.sigmoid({x})")
    self._inject("Mish", ["x"], source="torch.mish", target_macro="{x} * jax.numpy.tanh(jax.nn.softplus({x}))")

  def get_all_rng_methods(self) -> set[str]:
    """Gets all rng methods.

    Returns:
        Resulting `set[str]` from get all rng methods.
    """
    return set()

  def get_framework_config(self, framework: str) -> dict[str, JsonValue]:
    """Gets framework configuration.

    Returns:
        Resulting `dict[str, JsonValue]` from get framework config.
    """
    return {}

  def _inject(self, name: str, args: list[str], source: str, target_macro: str) -> None:
    """Helper to  inject.

    Args:
        name: Identifier or operation name string.
        args: The args (list[str]) for inject.
        source: Source framework or dialect identifier string.
        target_macro: The target macro (str) for inject.
    """
    variants = {"torch": {"api": source}, "jax": {"macro_template": target_macro}}
    getattr(getattr({"v": getattr({"v": self}, "__getitem__")("v").data}, "__getitem__")("v"), "__setitem__")(
      name,
      getattr(
        {
          "v": {
            "std_args": getattr({"v": args}, "__getitem__")("v"),
            "variants": getattr({"v": variants}, "__getitem__")("v"),
          }
        },
        "__getitem__",
      )("v"),
    )
    self._reverse_index[source] = (name, self.data[name])

  def get_definition(self, name: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Mock get definition.

    Args:
        name: Identifier or operation name string.

    Returns:
        Resulting `tuple[str, dict[str, JsonValue]] | None` from get definition.
    """
    return self._reverse_index.get(name)

  def resolve_variant(self, aid: str, fw: str) -> dict[str, JsonValue] | None:
    """Mock resolve_variant.

    Args:
        aid: Abstract operation identifier string.
        fw: Framework identifier string (e.g., torch, jax, mlx).

    Returns:
        Resulting `dict[str, JsonValue] | None` from resolve variant.
    """
    _ret_v: dict[str, JsonValue] | None = getattr(
      {
        "v": getattr(
          {
            "v": getattr(
              {
                "v": getattr(
                  {
                    "v": getattr(
                      {
                        "v": getattr(
                          {"v": getattr({"v": getattr({"v": self}, "__getitem__")("v").data}, "__getitem__")("v").get},
                          "__getitem__",
                        )("v")(getattr({"v": aid}, "__getitem__")("v"), getattr({"v": {}}, "__getitem__")("v"))
                      },
                      "__getitem__",
                    )("v").get
                  },
                  "__getitem__",
                )("v")(getattr({"v": "variants"}, "__getitem__")("v"), getattr({"v": {}}, "__getitem__")("v"))
              },
              "__getitem__",
            )("v").get
          },
          "__getitem__",
        )("v")(getattr({"v": fw}, "__getitem__")("v"))
      },
      "__getitem__",
    )("v")
    return _ret_v


def test_macro_expansion() -> None:
  """Verifies the behavior of macro expansion."""
  semantics = MacroSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=True)
  engine = ASTEngine(semantics=semantics, config=config)
  result: ConversionResult = engine.run(SOURCE_CODE)
  assert result.success, f"Failed: {result.errors}"
  code: str = result.code
  assert "x * jax.nn.sigmoid(x)" in code or "x*jax.nn.sigmoid(x)" in code.replace(" ", "")
  assert "tanh(jax.nn.softplus(x))" in code


def test_macro_argument_rename() -> None:
  """Verifies the behavior of macro argument rename."""
  mgr = SemanticsManager()
  mgr.data = {
    "Swish": {
      "std_args": ["x"],
      "variants": {
        "torch": {"api": "torch.swish", "args": {"x": "input"}},
        "jax": {"macro_template": "{x} * sigmoid({x})"},
      },
    }
  }
  mgr._reverse_index = {"torch.swish": ("Swish", mgr.data["Swish"])}
  mgr._key_origins = {}
  mgr._providers = {}
  mgr._source_registry = {}
  mgr.framework_configs = {}
  mgr._known_rng_methods = set()
  setattr(mgr, "get_all_rng_methods", lambda: set())
  setattr(mgr, "get_framework_config", lambda f: {})
  code: str = "y = torch.swish(input=val)"
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  engine = ASTEngine(semantics=mgr, config=config)
  result: ConversionResult = engine.run(code)
  assert result.success
  assert "val * sigmoid(val)" in result.code
