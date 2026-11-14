"""Test suite for the E2E Examples module."""

import typing
from pathlib import Path

import pytest
from ml_switcheroo_ir.schema.ghost import SemanticTier
from pydantic import JsonValue

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.engine import ASTEngine, ConversionResult
from ml_switcheroo.enums import RuntimeTensorOrValue
from ml_switcheroo.semantics.manager import SemanticsManager

EXAMPLES_DIR: Path = Path(__file__).parent.parent.parent / "tests" / "examples"


def _read_code(filename: str) -> str:
  """Execute `path.is_file` to support read code in e2e examples tests.

  Args:
      filename: Input `filename` parameter value (`str`).

  Returns:
      Resulting `str` from read code.
  """
  path = EXAMPLES_DIR / filename
  if not path.is_file():
    pytest.fail(f"Example file not found: {path}")
  return path.read_text(encoding="utf-8")


class E2ESemantics(SemanticsManager):
  """Test double `E2ESemantics` used in e2e examples tests."""

  def __init__(self) -> None:
    """Initializes the E2ESemantics instance."""
    super().__init__(load_defaults=False)
    self.framework_configs = {
      "flax_nnx": {
        "alias": {"module": "flax.nnx", "name": "nnx"},
        "traits": {
          "module_base": "flax.nnx.Module",
          "forward_method": "__call__",
          "inject_magic_args": [["rngs", "flax.nnx.Rngs"]],
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
    self.import_data: dict[str, RuntimeTensorOrValue] = {}
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
    self._alias("jnp.abs", "abs")
    self._alias("jnp.mean", "mean")
    self._alias("nn.Linear", "Linear")
    self._alias("nn.Module", "Module")

  def get_definition(self, name: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Mock get definition.

    Args:
        name: Identifier or operation name string.

    Returns:
        Resulting `tuple[str, dict[str, JsonValue]] | None` from get definition.
    """
    return self._reverse_index.get(name)

  def resolve_variant(self, aid: str, fw: str) -> dict[str, JsonValue] | None:
    """Mock resolve variant.

    Args:
        aid: Abstract operation identifier string.
        fw: Framework identifier string (e.g., torch, jax, mlx).

    Returns:
        Resulting `dict[str, JsonValue] | None` from resolve variant.
    """
    variants = self.data.get(aid, {}).get("variants")
    if not isinstance(variants, dict):
      return None
    fw_variant = variants.get(fw)
    return fw_variant if isinstance(fw_variant, dict) else None

  def get_all_rng_methods(self) -> set[str]:
    """Gets all rng methods.

    Returns:
        Resulting `set[str]` from get all rng methods.
    """
    return self._known_rng_methods

  def get_framework_config(self, framework: str) -> dict[str, JsonValue]:
    """Gets framework configuration.

    Args:
        framework: Framework identifier string.

    Returns:
        Resulting `dict[str, JsonValue]` from get framework config.
    """
    return self.framework_configs.get(framework, {})

  def get_framework_aliases(self) -> dict[str, tuple[str, str]]:
    """Gets framework aliases.

    Returns:
        Resulting `dict[str, tuple[str, str]]` from get framework aliases.
    """
    aliases: dict[str, tuple[str, str]] = {}
    for fw, cfg in self.framework_configs.items():
      alias_cfg = cfg.get("alias")
      if isinstance(alias_cfg, dict):
        mod = alias_cfg.get("module")
        name = alias_cfg.get("name")
        if isinstance(mod, str) and isinstance(name, str):
          aliases[fw] = (mod, name)
    return aliases

  def get_import_map(self, target_framework: str) -> dict[str, tuple[str, str | None, str | None]]:
    """Gets import map.

    Args:
        target_framework: Input `target_framework` parameter value (`str`).

    Returns:
        Resulting `dict[str, tuple[str, str | None, str | None]]` from get import map.
    """
    result: dict[str, tuple[str, str | None, str | None]] = {}
    target_providers = self._providers.get(target_framework, {})
    for src_path, (_src_fw, tier) in self._source_registry.items():
      if tier in target_providers:
        conf = target_providers[tier]
        root = conf.get("root")
        if isinstance(root, str):
          result[src_path] = (
            root,
            conf.get("sub"),
            conf.get("alias"),
          )
    return result

  def _add_op(self, name: str, args: list[str], tier: SemanticTier | None = None, **variants: str) -> None:
    """Execute `variants.items` to support add operation in e2e examples tests.

    Args:
        name: Identifier or operation name string.
        args: Positional argument specifications (`list[str]`).
        tier: Input `tier` parameter value (`SemanticTier | None`).
        variants: Per-framework operation variant mappings (`str`).
    """
    variant_data: dict[str, JsonValue] = {}
    std_args: list[JsonValue] = list(args)
    self.data[name] = {
      "std_args": std_args,
      "variants": variant_data,
    }
    for fw, api in variants.items():
      variant_data[fw] = {"api": api}
      self._alias(api, name)
    if tier:
      self._key_origins[name] = tier.value
    elif name[0].isupper():
      self._key_origins[name] = SemanticTier.NEURAL.value
    else:
      self._key_origins[name] = SemanticTier.ARRAY_API.value

  def _alias(self, api_str: str, abstract_name: str) -> None:
    """Execute `_alias` on `E2ESemantics` for e2e examples tests.

    Args:
        api_str: Input `api_str` parameter value (`str`).
        abstract_name: Input `abstract_name` parameter value (`str`).
    """
    if abstract_name in self.data:
      self._reverse_index[api_str] = (abstract_name, self.data[abstract_name])


@pytest.fixture
def engine_factory() -> typing.Callable[[str, str, bool], ASTEngine]:
  """Create a configured `engine_factory` fixture for e2e examples tests.

  Returns:
      Configured `engine_factory` fixture value (typing.Callable[[str, str, bool], ASTEngine]).
  """
  semantics = E2ESemantics()

  def _create(source: str, target: str, strict: bool = False) -> ASTEngine:
    """Execute `RuntimeConfig` to support create in e2e examples tests.

    Args:
        source: Source framework or dialect identifier string.
        target: Target framework or dialect identifier string.
        strict: Input `strict` parameter value (`bool`).

    Returns:
        Resulting `ASTEngine` from create.
    """
    cfg = RuntimeConfig(source_framework=source, target_framework=target, strict_mode=strict)
    return ASTEngine(semantics=semantics, config=cfg)

  return _create


def test_ex01_math_ops_torch_to_jax(engine_factory: typing.Callable[[str, str, bool], ASTEngine]) -> None:
  """Verify `_read_code` handles ex01 math operations torch to JAX as expected.

  Args:
      engine_factory: Input `engine_factory` parameter value (`typing.Callable[[str, str, bool], ASTEngine]`).
  """
  code: str = _read_code("ex01_math_ops.torch.py")
  engine = engine_factory("torch", "jax", False)
  result: ConversionResult = engine.run(code)
  assert result.success
  assert "import jax.numpy as jnp" in result.code
  assert "jnp.abs" in result.code


def test_ex01_math_ops_jax_to_torch(engine_factory: typing.Callable[[str, str, bool], ASTEngine]) -> None:
  """Verify `_read_code` handles ex01 math operations JAX to torch as expected.

  Args:
      engine_factory: Input `engine_factory` parameter value (`typing.Callable[[str, str, bool], ASTEngine]`).
  """
  code: str = _read_code("ex01_math_ops.jax.py")
  engine = engine_factory("jax", "torch", False)
  result: ConversionResult = engine.run(code)
  assert result.success
  assert "import torch" in result.code
  assert "torch.abs" in result.code


def test_ex02_neural_net_torch_to_jax(engine_factory: typing.Callable[[str, str, bool], ASTEngine]) -> None:
  """Verify `_read_code` emits `import flax.nnx as nnx` for ex02 neural net torch to JAX.

  Args:
      engine_factory: Input `engine_factory` parameter value (`typing.Callable[[str, str, bool], ASTEngine]`).
  """
  code: str = _read_code("ex02_neural_net.torch.py")
  engine = engine_factory("torch", "flax_nnx", False)
  result: ConversionResult = engine.run(code)
  assert result.success
  assert "import flax.nnx as nnx" in result.code
  assert "class SimplePerceptron(nnx.Module):" in result.code


def test_ex02_neural_net_jax_to_torch(engine_factory: typing.Callable[[str, str, bool], ASTEngine]) -> None:
  """Verify `_read_code` handles ex02 neural net JAX to torch as expected.

  Args:
      engine_factory: Input `engine_factory` parameter value (`typing.Callable[[str, str, bool], ASTEngine]`).
  """
  code: str = _read_code("ex02_neural_net.flax_nnx.py")
  engine = engine_factory("flax_nnx", "torch", False)
  result: ConversionResult = engine.run(code)
  assert result.success
  assert "import torch.nn as nn" in result.code or "from torch import nn" in result.code
  assert "class SimplePerceptron(nn.Module):" in result.code
