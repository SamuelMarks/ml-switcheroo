"""Test suite for the Flax Torch Bidirectional module."""

import ast

import pytest
from ml_switcheroo_ir.schema.ghost import SemanticTier
from pydantic import JsonValue

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.engine import ASTEngine, ConversionResult
from ml_switcheroo.core.escape_hatch import EscapeHatch
from ml_switcheroo.enums import RuntimeTensorOrValue
from ml_switcheroo.frameworks.flax_nnx import FlaxNNXAdapter
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.semantics.merging import merge_overlay_data
from tests.utils.ast_utils import cmp_ast

flax_nnx_tier2_ex0: str = "\nfrom flax import nnx\n\nclass Net(nnx.Module):\n    def __init__(self, rngs: nnx.Rngs):\n        # State injection pattern\n        self.linear = nnx.Linear(10, 10, rngs=rngs)\n\n    def __call__(self, x):\n        x = self.linear(x)\n        # Functional activation\n        return nnx.relu(x)\n"
torch_tier2_ex0: str = "\nimport torch.nn.functional as F\nfrom torch import nn\n\nclass Net(nn.Module):\n    def __init__(self):\n        super().__init__()\n        # State injection pattern\n        self.linear = nn.Linear(10, 10)\n\n    def forward(self, x):\n        x = self.linear(x)\n        # Functional activation\n        return F.relu(x)\n"


@pytest.fixture(scope="module")
def semantics() -> SemanticsManager:
  """Create a configured `SemanticsManager` fixture for flax torch bidirectional tests.

  Returns:
      Configured `semantics` fixture value (SemanticsManager).
  """
  return SemanticsManager()


def check_mappings_exist(semantics: SemanticsManager) -> None:
  """Checks mappings exist.

  Args:
      semantics: SemanticsManager instance or mock providing operation mappings.
  """
  lin_def = semantics.get_definition_by_id("Linear")
  lin_variants = lin_def.get("variants") if isinstance(lin_def, dict) else None
  if not isinstance(lin_variants, dict) or "torch" not in lin_variants:
    pytest.skip("Missing 'Linear' mapping in Knowledge Base. Run `./scripts/bootstrap.sh`")
  relu_def = semantics.get_definition_by_id("relu")
  relu_variants = relu_def.get("variants") if isinstance(relu_def, dict) else None
  if (not isinstance(relu_variants, dict) or "torch" not in relu_variants) and not semantics.get_definition_by_id("ReLU"):
    pytest.skip("Missing 'relu/ReLU' mapping in Knowledge Base. Run: `./scripts/bootstrap.sh`")


@pytest.mark.skip(reason="ReLU mapping removed from nnx")
def test_flax_nnx_to_torch_neural_ex0(semantics: SemanticsManager) -> None:
  """Verify `check_mappings_exist` emits `class Net(nn.Module):` for flax NNX to torch neural ex0.

  Args:
      semantics: SemanticsManager instance or mock providing operation mappings.
  """
  check_mappings_exist(semantics)
  result: ConversionResult = ASTEngine(
    semantics=semantics, config=RuntimeConfig(source_framework="flax_nnx", target_framework="torch", strict_mode=True)
  ).run(flax_nnx_tier2_ex0)
  print(f"\n[Generated Code]:\n{result.code}")
  assert EscapeHatch.START_MARKER not in result.code, f"Escape Hatch detected. Semantics missing? Errors: {result.errors}"
  try:
    assert cmp_ast(ast.parse(result.code), ast.parse(torch_tier2_ex0))
  except (SyntaxError, AssertionError):
    assert "class Net(nn.Module):" in result.code
    assert "super().__init__()" in result.code
    assert "nn.Linear(10, 10)" in result.code
    if "F.relu(x)" not in result.code and "torch.nn.functional.relu(x)" not in result.code:
      if "nn.ReLU(x)" in result.code:
        pytest.fail("Generated nn.ReLU(x) (Class Instantiation) instead of F.relu(x) (Functional Call).")
      else:
        pytest.fail(f"Missing F.relu(x). Code:\n{result.code}")
    assert "def forward(self, x):" in result.code


class FixedSemantics(SemanticsManager):
  """Minimal `SemanticsManager` test double used in flax torch bidirectional tests."""

  def __init__(self) -> None:
    """Initializes the FixedSemantics instance."""
    super().__init__(load_defaults=False)
    self.import_data: dict[str, RuntimeTensorOrValue] = {}
    adapter = FlaxNNXAdapter()
    snapshot: dict[str, dict[str, JsonValue]] = {}
    adapter.apply_wiring(snapshot)
    overlay: dict[str, JsonValue] = {"__framework__": "jax", "mappings": {}, "imports": {}, **snapshot}
    merge_overlay_data(
      data=self.data,
      key_origins=self._key_origins,
      framework_configs=self.framework_configs,
      test_templates=self.test_templates,
      content=overlay,
      filename="jax_vlatest_map.json",
    )
    self.data["Abs"] = {"std_args": ["x"], "variants": {"torch": {"api": "torch.abs"}, "jax": {"api": "jax.numpy.abs"}}}
    self.data["Module"] = {"std_args": [], "variants": {"torch": {"api": "torch.nn.Module"}}}
    self.framework_configs["jax"] = {"traits": adapter.structural_traits.model_dump(exclude_unset=True)}
    self.framework_configs["torch"] = {"traits": {"module_base": "torch.nn.Module", "forward_method": "forward"}}
    self.framework_configs["jax"]["alias"] = {"module": "jax.numpy", "name": "jnp"}
    self._key_origins["Abs"] = SemanticTier.NEURAL.value
    self._key_origins["Module"] = SemanticTier.NEURAL.value
    self._build_index()
    self._source_registry["torch.nn"] = ("torch", SemanticTier.NEURAL)
    if "jax" not in self._providers:
      self._providers["jax"] = {}
    self._providers["jax"][SemanticTier.NEURAL] = {"root": "flax", "sub": "nnx", "alias": "nnx"}


def test_specific_abs_conversion() -> None:
  """Verify `FixedSemantics` emits `import jax.numpy as jnp` for specific abs conversion."""
  input_torch: str = "\nimport torch\nimport torch.nn as nn\n\nclass Model(nn.Module):\n    def forward(self, x):\n        return torch.abs(x)\n"
  semantics = FixedSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=False)
  engine = ASTEngine(semantics=semantics, config=config)
  result: ConversionResult = engine.run(input_torch)
  assert result.success
  code: str = result.code
  assert "import jax.numpy as jnp" in code
  assert "import flax.nnx as nnx" in code or "from flax import nnx" in code
  assert "import torch" not in code
  assert "as nn" not in code.split("\n")[1:]
  assert "class Model(nnx.Module):" in code
  assert "def __call__(self, x):" in code
  assert "jnp.abs(x)" in code
