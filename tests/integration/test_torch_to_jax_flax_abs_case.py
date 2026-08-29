"""Test suite for the Torch To Jax Flax Abs Case module."""

import typing

from ml_switcheroo_ir.schema.ghost import SemanticTier

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.engine import ASTEngine, ConversionResult
from ml_switcheroo.frameworks.flax_nnx import FlaxNNXAdapter
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.semantics.merging import merge_overlay_data


class FixedSemantics(SemanticsManager):  # type: ignore[misc]
  """Docstring."""

  def __init__(self) -> None:
    """Initializes the FixedSemantics instance."""
    self.data: dict[str, typing.Any] = {}
    self.framework_configs: dict[str, typing.Any] = {}
    self.test_templates: dict[str, typing.Any] = {}
    self._known_rng_methods: set[str] = set()
    self.known_magic_args: set[str] = set()
    self.patterns: list[typing.Any] = []
    self._reverse_index: dict[str, tuple[str, dict[str, typing.Any]]] = {}
    self._key_origins: dict[str, str] = {}
    self._validation_status: dict[str, typing.Any] = {}
    self._providers: dict[str, typing.Any] = {}
    self._source_registry: dict[str, typing.Any] = {}
    adapter = FlaxNNXAdapter()
    snapshot: dict[str, typing.Any] = {"__framework__": "jax", "mappings": {}, "imports": {}}
    adapter.apply_wiring(snapshot)
    merge_overlay_data(
      data=self.data,
      key_origins=self._key_origins,
      framework_configs=self.framework_configs,
      test_templates=self.test_templates,
      content=snapshot,
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
  """Verifies the behavior of specific abs conversion."""
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
