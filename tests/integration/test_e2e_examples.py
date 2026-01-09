"""End-to-End Integration Tests."""

import pytest
from pathlib import Path
from typing import Set, Dict, Tuple, Optional

from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.enums import SemanticTier
from ml_switcheroo.core.escape_hatch import EscapeHatch

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


def _read_code(filename: str) -> str:
  path = EXAMPLES_DIR / filename
  if not path.is_file():
    pytest.fail(f"Example file not found: {path}")
  return path.read_text(encoding="utf-8")


class E2ESemantics(SemanticsManager):
  def __init__(self):
    self.data = {}
    self._providers = {}
    self._source_registry = {}

    # Fix: Added 'jax' and 'tensorflow' config to enable alias resolution and traits
    self.framework_configs = {
      "flax_nnx": {
        "alias": {"module": "flax.nnx", "name": "nnx"},
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
      "jax": {"alias": {"module": "jax.numpy", "name": "jnp"}},
      "tensorflow": {"alias": {"module": "tensorflow", "name": "tf"}},
    }
    self._reverse_index = {}
    self._key_origins = {}
    self._validation_status = {}
    self._known_rng_methods = set()

    self._add_op("abs", ["x"], torch="torch.abs", jax="jax.numpy.abs", keras="keras.ops.abs")
    self._add_op("mean", ["x"], torch="torch.mean", jax="jax.numpy.mean", keras="keras.ops.mean")
    self._add_op("sub", ["x", "y"], torch="torch.sub", jax="jax.numpy.subtract")
    self._add_op(
      "Module", [], torch="torch.nn.Module", jax="flax.nnx.Module", keras="keras.Model", tier=SemanticTier.NEURAL
    )
    self._add_op(
      "Linear",
      ["in_features", "out_features"],
      torch="torch.nn.Linear",
      jax="flax.nnx.Linear",
      keras="keras.layers.Dense",
      tier=SemanticTier.NEURAL,
    )

    self._source_registry["torch.nn"] = ("torch", SemanticTier.NEURAL)
    self._source_registry["flax.nnx"] = ("flax_nnx", SemanticTier.NEURAL)
    self._source_registry["jax.numpy"] = ("jax", SemanticTier.ARRAY_API)
    self._source_registry["jnp"] = ("jax", SemanticTier.ARRAY_API)
    self._source_registry["torch"] = ("torch", SemanticTier.ARRAY_API)

    self._providers.setdefault("flax_nnx", {})[SemanticTier.NEURAL] = {"root": "flax", "sub": "nnx", "alias": "nnx"}
    self._providers.setdefault("torch", {})[SemanticTier.NEURAL] = {"root": "torch", "sub": "nn", "alias": "nn"}

    # Fix: Add JAX/TF providers
    self._providers.setdefault("jax", {})[SemanticTier.ARRAY_API] = {"root": "jax", "sub": "numpy", "alias": "jnp"}
    self._providers.setdefault("tensorflow", {})[SemanticTier.ARRAY_API] = {
      "root": "tensorflow",
      "sub": None,
      "alias": "tf",
    }

    self._add_op(
      "transpose", ["x", "axes"], torch="torch.permute", jax="jax.numpy.transpose", keras="keras.ops.transpose"
    )
    self._add_op(
      "flatten_func", ["x", "start_dim"], torch="torch.flatten", jax="jax.numpy.ravel", keras="keras.ops.ravel"
    )
    self._add_op("Conv2d", ["in", "out", "k"], torch="torch.nn.Conv2d", jax="flax.nnx.Conv", keras="keras.layers.Conv2D")
    self._add_op("Dropout", ["p"], torch="torch.nn.Dropout", jax="flax.nnx.Dropout", keras="keras.layers.Dropout")
    self._add_op("Flatten", [], torch="torch.nn.Flatten", jax="flax.nnx.Flatten", keras="keras.layers.Flatten")
    self._add_op(
      "MaxPool2d", ["k"], torch="torch.nn.MaxPool2d", jax="flax.nnx.max_pool", keras="keras.layers.MaxPooling2D"
    )
    self._add_op("Input", ["shape"], torch="torch.empty", jax="jax.numpy.empty", keras="keras.Input")
    self._add_op("relu", ["x"], torch="torch.nn.functional.relu", jax="jax.nn.relu")
    self._add_op("relu_f", ["x"], torch="torch.relu", jax="jax.nn.relu")
    self._add_op("log_softmax", ["x"], torch="torch.nn.functional.log_softmax", jax="jax.nn.log_softmax")
    self._add_op("max_pool2d_func", ["x"], torch="torch.nn.functional.max_pool2d", jax="jax.lax.reduce_window")

    # Fix: Manually map 'jnp' to these alias roots for reverse lookup if needed
    self._alias("jnp.abs", "abs")
    self._alias("jnp.mean", "mean")
    self._alias("nn.Linear", "Linear")
    self._alias("nn.Module", "Module")
    # ... more aliases omitted for brevity

  def get_all_rng_methods(self) -> Set[str]:
    return self._known_rng_methods

  def get_framework_config(self, framework: str):
    return self.framework_configs.get(framework, {})

  def get_framework_aliases(self) -> Dict[str, Tuple[str, str]]:
    # Mock the alias getter to return what we populated in framework_configs
    aliases = {}
    for fw, cfg in self.framework_configs.items():
      if "alias" in cfg:
        aliases[fw] = (cfg["alias"]["module"], cfg["alias"]["name"])
    return aliases

  def get_import_map(self, target_fw: str) -> Dict[str, Tuple[str, Optional[str], Optional[str]]]:
    result = {}
    target_providers = self._providers.get(target_fw, {})
    for src_path, (src_fw, tier) in self._source_registry.items():
      if tier in target_providers:
        conf = target_providers[tier]
        result[src_path] = (conf["root"], conf["sub"], conf["alias"])
    return result

  def _add_op(self, name, args, tier=None, **variants):
    variant_data = {}
    # 1. Update self.data first so _alias can reference it
    self.data[name] = {"std_args": args, "variants": variant_data}

    # 2. Iterate variants to populate data AND alias index
    for fw, api in variants.items():
      variant_data[fw] = {"api": api}
      self._alias(api, name)

    # 3. Set origins
    if tier:
      self._key_origins[name] = tier.value
    elif name[0].isupper():
      self._key_origins[name] = SemanticTier.NEURAL.value
    else:
      self._key_origins[name] = SemanticTier.ARRAY_API.value

  def _alias(self, api_str, abstract_name):
    if abstract_name in self.data:
      self._reverse_index[api_str] = (abstract_name, self.data[abstract_name])


@pytest.fixture
def engine_factory():
  semantics = E2ESemantics()

  def _create(source, target, strict=False):
    # Fix: Ensure source_flavour is handled if relevant, but basic tests use root
    cfg = RuntimeConfig(source_framework=source, target_framework=target, strict_mode=strict)
    return ASTEngine(semantics=semantics, config=cfg)

  return _create


def test_ex01_math_ops_torch_to_jax(engine_factory):
  code = _read_code("ex01_math_ops.torch.py")
  engine = engine_factory("torch", "jax")
  result = engine.run(code)
  assert result.success
  assert "import jax.numpy as jnp" in result.code
  assert "jnp.abs" in result.code


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
  assert result.success
  assert "import flax.nnx as nnx" in result.code
  assert "class SimplePerceptron(nnx.Module):" in result.code


def test_ex02_neural_net_jax_to_torch(engine_factory):
  code = _read_code("ex02_neural_net.flax_nnx.py")
  engine = engine_factory("flax_nnx", "torch")
  result = engine.run(code)
  assert result.success

  # Allow both valid import forms
  assert "import torch.nn as nn" in result.code or "from torch import nn" in result.code
  assert "class SimplePerceptron(nn.Module):" in result.code
  assert "def forward(self, x):" in result.code
  assert "def __init__(self, in_features, out_features):" in result.code
