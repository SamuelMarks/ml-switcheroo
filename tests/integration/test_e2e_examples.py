"""End-to-End Integration Tests for Example Files."""

import pytest
from pathlib import Path
from typing import Set, Dict, Tuple, Optional

from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.enums import SemanticTier
from ml_switcheroo.core.escape_hatch import EscapeHatch

# Resolve path relative to this test file
EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


def _read_code(filename: str) -> str:
  path = EXAMPLES_DIR / filename
  if not path.is_file():
    pytest.fail(f"Example file not found: {path}")
  return path.read_text(encoding="utf-8")


class E2ESemantics(SemanticsManager):
  def __init__(self):
    # Initialize empty stores manually
    self.data = {}
    # New attributes replacing import_data
    self._providers = {}
    self._source_registry = {}

    self.framework_configs = {
      "flax_nnx": {
        "alias": {"module": "flax.nnx", "name": "nnx"},  # Set Default Alias for tests
        "traits": {
          "module_base": "flax.nnx.Module",
          "forward_method": "__call__",
          "inject_magic_args": [("rngs", "flax.nnx.Rngs")],
        },
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

    self._reverse_index = {}
    self._key_origins = {}
    self._validation_status = {}
    self._known_rng_methods = set()

    # 1. Math Operations (ex01)
    self._add_op("abs", ["x"], torch="torch.abs", jax="jax.numpy.abs")
    self._add_op("mean", ["x"], torch="torch.mean", jax="jax.numpy.mean")
    self._add_op("sub", ["x", "y"], torch="torch.sub", jax="jax.numpy.subtract")

    # 2. Neural Networks (ex02)
    self._add_op("Module", [], torch="torch.nn.Module", jax="flax.nnx.Module", tier=SemanticTier.NEURAL)
    self._add_op(
      "Linear",
      ["in_features", "out_features"],
      torch="torch.nn.Linear",
      jax="flax.nnx.Linear",
      tier=SemanticTier.NEURAL,
    )

    # Import Remapping setup (New Architecture)
    # Register Source Paths
    self._source_registry["torch.nn"] = ("torch", SemanticTier.NEURAL)
    self._source_registry["flax.nnx"] = ("flax_nnx", SemanticTier.NEURAL)

    # Register Providers
    # Target: flax_nnx (Neural) -> root: flax, sub: nnx, alias: nn
    self._providers.setdefault("flax_nnx", {})[SemanticTier.NEURAL] = {
      "root": "flax",
      "sub": "nnx",
      "alias": "nn",
    }

    # Target: torch (Neural) -> root: torch, sub: nn, alias: nn
    self._providers.setdefault("torch", {})[SemanticTier.NEURAL] = {
      "root": "torch",
      "sub": "nn",
      "alias": "nn",
    }

    # 3. Array Manipulation
    self._add_op("transpose", ["x", "axes"], torch="torch.permute", jax="jax.numpy.transpose")

    # Aliases
    self._alias("jnp.abs", "abs")
    self._alias("jnp.mean", "mean")
    self._alias("jnp.transpose", "transpose")
    self._alias("nn.Module", "Module")
    self._alias("nnx.Module", "Module")
    self._alias("nnx.Linear", "Linear")
    self._alias("nn.Linear", "Linear")

  def get_all_rng_methods(self) -> Set[str]:
    return self._known_rng_methods

  def get_framework_config(self, framework: str):
    return self.framework_configs.get(framework, {})

  def get_import_map(self, target_fw: str) -> Dict[str, Tuple[str, Optional[str], Optional[str]]]:
    # Mocking get_import_map for test stability
    result = {}
    target_providers = self._providers.get(target_fw, {})

    for src_path, (src_fw, tier) in self._source_registry.items():
      if tier in target_providers:
        conf = target_providers[tier]
        result[src_path] = (conf["root"], conf["sub"], conf["alias"])
    return result

  def _add_op(self, name, args, tier=None, **variants):
    """Helper to define a multi-way op."""
    variant_data = {}

    # 1. Update data first so it exists for reverse index lookup
    self.data[name] = {"std_args": args, "variants": variant_data}

    for fw, api in variants.items():
      variant_data[fw] = {"api": api}
      # Register reverse lookup for this specific API path
      self._alias(api, name)

    # Ensure variants structure is complete in the stored dict ref
    self.data[name]["variants"] = variant_data

    # Set tier origins
    if tier:
      self._key_origins[name] = tier.value
    elif name[0].isupper():
      self._key_origins[name] = SemanticTier.NEURAL.value
    else:
      self._key_origins[name] = SemanticTier.ARRAY_API.value

  def _alias(self, api_str, abstract_name):
    """Map a specific string string to an abstract definition."""
    if abstract_name in self.data:
      self._reverse_index[api_str] = (abstract_name, self.data[abstract_name])


@pytest.fixture
def engine_factory():
  semantics = E2ESemantics()

  def _create(source, target, strict=False):
    cfg = RuntimeConfig(source_framework=source, target_framework=target, strict_mode=strict)
    return ASTEngine(semantics=semantics, config=cfg)

  return _create


# Tests are largely same, updated assertion strings to match aliased output


def test_ex01_math_ops_torch_to_jax(engine_factory):
  code = _read_code("ex01_math_ops.torch.py")
  engine = engine_factory("torch", "jax")
  result = engine.run(code)
  assert result.success
  assert (
    "import jax" in result.code
    or "from jax import numpy as jnp" in result.code
    or "import jax.numpy as jnp" in result.code
  )
  assert "jax.numpy.abs" in result.code or "jnp.abs" in result.code


def test_ex01_math_ops_jax_to_torch(engine_factory):
  code = _read_code("ex01_math_ops.jax.py")
  engine = engine_factory("jax", "torch")
  result = engine.run(code)
  assert result.success
  assert "import torch" in result.code
  assert "torch.abs" in result.code


def test_ex02_neural_net_torch_to_jax(engine_factory):
  code = _read_code("ex02_neural_net.torch.py")
  engine = engine_factory("torch", "flax_nnx")
  result = engine.run(code)

  if not result.success:
    pytest.fail(f"Conversion failed: {result.errors}")

  assert "from flax import nnx" in result.code or "import flax.nnx as nnx" in result.code

  # Assert Aliased usage since "import flax.nnx as nn" is used
  # Assuming the provider configuration 'alias': 'nn' takes precedence
  assert "class SimplePerceptron(nn.Module):" in result.code or "class SimplePerceptron(nnx.Module):" in result.code
  assert "nn.Linear" in result.code or "nnx.Linear" in result.code
  assert "rngs=rngs" in result.code
  assert "def __call__(self, x):" in result.code


def test_ex02_neural_net_jax_to_torch(engine_factory):
  code = _read_code("ex02_neural_net.flax_nnx.py")
  engine = engine_factory("flax_nnx", "torch")
  result = engine.run(code)

  if not result.success:
    pytest.fail(f"Conversion failed: {result.errors}")

  # The import fixer will generate submodule import "from torch import nn" without suffix
  # OR "from torch import nn as nn" based on logic in import_fixer
  # The redundancy check should reduce "as nn" so we check for both legitimate output forms
  assert "from torch import nn" in result.code or "import torch.nn as nn" in result.code
  assert "class SimplePerceptron(nn.Module):" in result.code
  assert "def forward(self, x):" in result.code
  assert "super().__init__()" in result.code
  assert "def __init__(self, in_features, out_features):" in result.code


def test_ex03_array_manip_torch_to_jax(engine_factory):
  code = _read_code("ex03_array_manip.torch.py")
  engine = engine_factory("torch", "jax")
  result = engine.run(code)
  assert result.success
  assert "jax.numpy.transpose" in result.code


def test_ex03_array_manip_jax_to_torch(engine_factory):
  code = _read_code("ex03_array_manip.jax.py")
  engine = engine_factory("jax", "torch")
  result = engine.run(code)
  assert result.success
  assert "torch.permute" in result.code


def test_ex04_mixed_checkpointing_torch_to_jax(engine_factory):
  code = _read_code("ex04_mixed_checkpointing.torch.py")
  engine = engine_factory("torch", "jax", strict=True)
  result = engine.run(code)
  assert result.success
  assert result.has_errors
  assert "jax.numpy.abs" in result.code or "jnp.abs" in result.code
  assert EscapeHatch.START_MARKER in result.code
  assert "checkpoint.checkpoint(" in result.code


def test_ex05_mixed_parallelism_jax_to_torch(engine_factory):
  code = _read_code("ex05_mixed_parallelism.jax.py")
  engine = engine_factory("jax", "torch", strict=True)
  result = engine.run(code)
  assert result.success
  assert result.has_errors
  assert "torch.abs" in result.code
  assert EscapeHatch.START_MARKER in result.code
  assert "pmap(" in result.code
