"""Test suite for the Tensor Layout module."""

import typing

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.engine import ASTEngine, ConversionResult
from ml_switcheroo.semantics.manager import SemanticsManager

SOURCE_LAYOUT: str = (
  "\nimport torch\n\ndef process_image(x):\n    # Assume source is NCHW\n    return torch.conv2d(x, w)\n"
)
EXPECTED_OUTPUT: str = "\nimport jax.numpy as jnp\n\ndef process_image(x):\n    # Assume source is NCHW\n    return jnp.transpose(jax.lax.conv(jnp.transpose(x, axes=(0, 2, 3, 1)), w), axes=(0, 3, 1, 2))\n"


class LayoutSemantics(SemanticsManager):  # type: ignore[misc]
  """Docstring."""

  def __init__(self) -> None:
    """Initializes the LayoutSemantics instance."""
    self.data: dict[str, typing.Any] = {}
    self._providers: dict[str, typing.Any] = {}
    self._source_registry: dict[str, typing.Any] = {}
    self.import_data: dict[str, typing.Any] = {}
    self.framework_configs: dict[str, typing.Any] = {}
    self._reverse_index: dict[str, tuple[str, dict[str, typing.Any]]] = {}
    self._key_origins: dict[str, str] = {}
    self._validation_status: dict[str, typing.Any] = {}
    self._known_rng_methods: set[str] = set()
    self.data["Conv2d"] = {
      "std_args": ["input", "weight"],
      "variants": {
        "torch": {"api": "torch.conv2d"},
        "jax": {
          "api": "jax.lax.conv",
          "args": {"input": "lhs", "weight": "rhs"},
          "layout_map": {"input": "NCHW->NHWC", "return": "NHWC->NCHW"},
        },
      },
    }
    self._reverse_index["torch.conv2d"] = ("Conv2d", self.data["Conv2d"])
    self.data["permute_dims"] = {
      "std_args": ["x", "axes"],
      "variants": {"jax": {"api": "jnp.transpose", "pack_to_tuple": "axes"}},
    }

  def get_all_rng_methods(self) -> set[str]:
    """Gets all rng methods."""
    return set()


def test_layout_permutation_injection() -> None:
  """Verifies the behavior of layout permutation injection."""
  semantics = LayoutSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=False)
  engine = ASTEngine(semantics=semantics, config=config)
  result: ConversionResult = engine.run(SOURCE_LAYOUT)
  assert result.success
  code: str = result.code
  clean: str = code.replace(" ", "").replace("\n", "")
  assert "jnp.transpose(x,axes=(0,2,3,1))" in clean
  assert "jnp.transpose(jax.lax.conv" in clean
  assert ",axes=(0,3,1,2))" in clean
  assert "jax.lax.conv" in result.code
