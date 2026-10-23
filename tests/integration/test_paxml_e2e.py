"""Test suite for the Paxml E2E module."""

from ml_switcheroo.enums import RuntimeTensorOrValue

from pydantic import JsonValue
from pathlib import Path

import pytest
from ml_switcheroo_ir.schema.ghost import SemanticTier

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.engine import ASTEngine, ConversionResult
from ml_switcheroo.semantics.manager import SemanticsManager

EXAMPLES_DIR: Path = Path(__file__).parent.parent / "examples"


def _read_code(filename: str) -> str:
  """Helper to  read code.

  Args:
      filename: The filename (str) for  read code.

  Returns:
      Resulting `str` from read code.
  """
  path: Path = EXAMPLES_DIR / filename
  return path.read_text(encoding="utf-8")


class PaxE2ESemantics(SemanticsManager):
  """Test helper class PaxE2ESemantics."""

  def __init__(self) -> None:
    """Initializes the PaxE2ESemantics instance."""
    self.data = {}
    self._providers = {}
    self._source_registry = {}
    self.import_data: dict[str, RuntimeTensorOrValue] = {}
    self._reverse_index = {}
    self._key_origins: dict[str, str] = {}
    self._known_rng_methods: set[str] = set()
    self.framework_configs = {
      "paxml": {
        "traits": {
          "module_base": "praxis.base_layer.BaseLayer",
          "forward_method": "__call__",
          "init_method_name": "setup",
          "requires_super_init": False,
        }
      },
      "torch": {
        "traits": {
          "module_base": "torch.nn.Module",
          "forward_method": "forward",
          "strip_magic_args": ["rngs"],
          "requires_super_init": True,
        }
      },
    }
    self._add_op("Module", [], torch="torch.nn.Module", pax="praxis.base_layer.BaseLayer", tier=SemanticTier.NEURAL)
    self._add_op(
      "Linear",
      ["in_features", "out_features"],
      torch="torch.nn.Linear",
      pax="praxis.layers.Linear",
      tier=SemanticTier.NEURAL,
    )
    self._add_op("ReLU", [], torch="torch.nn.ReLU", pax="praxis.layers.ReLU", tier=SemanticTier.NEURAL)
    self._source_registry["torch.nn"] = ("torch", SemanticTier.NEURAL)
    if "paxml" not in self._providers:
      self._providers["paxml"] = {}
    self._providers["paxml"][SemanticTier.NEURAL] = {"root": "praxis", "sub": "layers", "alias": "nn"}
    self._alias("nn.Module", "Module")
    self._alias("nn.Linear", "Linear")
    self._alias("nn.ReLU", "ReLU")

  def get_all_rng_methods(self) -> set[str]:
    """Gets all rng methods.

    Returns:
        Resulting `set[str]` from get all rng methods.
    """
    return self._known_rng_methods

  def get_framework_config(self, framework: str) -> dict[str, JsonValue]:
    """Gets framework configuration.

    Returns:
        Configuration dictionary for the specified framework.
    """
    return self.framework_configs.get(framework, {})

  def _add_op(self, name: str, args: list[str], torch: str, pax: str, tier: SemanticTier | None = None) -> None:
    """Helper to  add op.

    Args:
        name: Identifier or operation name string.
        args: Positional arguments tuple or list.
        torch: The torch (str) for  add op.
        pax: The pax (str) for  add op.
        tier: The tier (SemanticTier | None) for  add op.
    """
    getattr(self.data, "__setitem__")(
      name,
      {
        "std_args": args,
        "variants": {
          "torch": {"api": torch},
          "paxml": {"api": pax},
        },
      },
    )
    if torch:
      self._reverse_index[torch] = (name, self.data[name])
    if pax:
      self._reverse_index[pax] = (name, self.data[name])
    if tier:
      self._key_origins[name] = tier.value
    else:
      self._key_origins[name] = SemanticTier.ARRAY_API.value

  def _alias(self, api_str: str, abstract_name: str) -> None:
    """Helper to  alias.

    Args:
        api_str: The api str (str) for  alias.
        abstract_name: The abstract name (str) for  alias.
    """
    if abstract_name in self.data:
      self._reverse_index[api_str] = (abstract_name, self.data[abstract_name])


@pytest.fixture
def pax_engine() -> ASTEngine:
  """Execute pax engine helper logic.

  Returns:
      Configured `pax_engine` fixture value (ASTEngine).
  """
  semantics = PaxE2ESemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="paxml", strict_mode=False)
  return ASTEngine(semantics=semantics, config=config)


def test_ex06_paxml_full_conversion(pax_engine: ASTEngine) -> None:
  """Verifies the behavior of ex06 Paxml full conversion.

  Args:
      pax_engine: The pax engine (ASTEngine) for test ex06 paxml full conversion.
  """
  code: str = _read_code("ex06_paxml.torch.py")
  result: ConversionResult = pax_engine.run(code)
  assert result.success, f"Conversion failed: {result.errors}"
  generated: str = result.code
  assert "import praxis" in generated or "from praxis" in generated
  assert "class SimpleMLP(praxis.base_layer.BaseLayer):" in generated
  assert "def setup(self, input_size" in generated
  assert "def __init__" not in generated
  assert "super().__init__()" not in generated
  assert "def __call__(self, x" in generated
  assert "def forward" not in generated
