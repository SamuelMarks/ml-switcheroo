"""Zero-Framework Isolated Sandbox and Traced Layer Wrapper for On-Device Execution.

Provides ``SandboxNamespace``, ``zero_framework_sandbox``, ``_resolve_traits``,
``_lookup_adapter``, and ``_TracedLayerInstance``.
"""

from __future__ import annotations

import contextlib
import importlib
import logging
import types
import typing

from ml_switcheroo.core.compiler.compiler_passes import ensure_compiler_and_zero_on_path
from ml_switcheroo.core.runtime_models import (
  ConstructorDimResolution,
  DeclaredLayerSpec,
)
from ml_switcheroo.core.runtime_tensor import (
  OnDeviceTraceTensor,
  _conv_nd,
  _generate_seeded_values,
  _matmul_2d,
  _prod,
)
from ml_switcheroo.frameworks.base import FrameworkAdapter
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.semantics.schema import StructuralTraits

_SUPPORT_ZERO_ALIASES: dict[str, str] = {
  "optax": "zero_optax",
  "chex": "zero_chex",
  "grain": "zero_grain",
  "orbax": "zero_orbax",
  "zero_zoo": "zero_zoo",
}


@typing.runtime_checkable
class _StructuralTraitsAdapterProtocol(typing.Protocol):
  """Protocol for adapters exposing ``structural_traits``."""

  @property
  def structural_traits(self) -> StructuralTraits:
    """Return structural traits.

    Returns:
        ``StructuralTraits`` instance.
    """
    ...


def _lookup_adapter(framework: str) -> FrameworkAdapter | None:
  """Look up ``framework`` adapter through ``ml_switcheroo.core.runtime_tracer.get_adapter``.

  Args:
      framework: Canonical framework identifier.

  Returns:
      Registered ``FrameworkAdapter`` instance, or ``None``.
  """
  from ml_switcheroo.core import runtime_tracer as _rt_facade

  return _rt_facade.get_adapter(framework)


def _resolve_traits(framework: str, semantics: SemanticsManager | None = None) -> StructuralTraits:
  """Resolve ``StructuralTraits`` for a framework key from its adapter or ``SemanticsManager``.

  Args:
      framework: Canonical framework identifier.
      semantics: Optional ``SemanticsManager`` instance.

  Returns:
      Resolved ``StructuralTraits`` instance.
  """
  adapter = _lookup_adapter(framework)
  if isinstance(adapter, _StructuralTraitsAdapterProtocol):
    traits = adapter.structural_traits
    if isinstance(traits, StructuralTraits):
      return traits
  if semantics is not None:
    fw_cfg = semantics.get_framework_config(framework)
    raw_traits = fw_cfg.get("traits")
    if isinstance(raw_traits, dict):
      return StructuralTraits.model_validate(raw_traits)
  return StructuralTraits()


class SandboxNamespace(dict[str, types.ModuleType]):
  """Thread-safe isolated execution namespace mapping framework paths to ``zero-*`` modules."""

  def resolve_import(self, module_name: str) -> types.ModuleType:
    """Resolve ``module_name`` against active ``zero-*`` sandbox bindings before standard imports.

    Args:
        module_name: Canonical dotted module name.

    Returns:
        Bound ``zero-*`` module or standard imported module.
    """
    if module_name in self:
      return self[module_name]
    return importlib.import_module(module_name)

  def build_exec_globals(
    self,
  ) -> dict[str, types.ModuleType | typing.Callable[[str], types.ModuleType]]:
    """Construct an isolated ``exec_globals`` mapping with sandbox root modules and ``__import_module__``.

    Returns:
        Isolated namespace dictionary.
    """
    ns: dict[str, types.ModuleType | typing.Callable[[str], types.ModuleType]] = {
      "__import_module__": self.resolve_import,
    }
    ns.update({canonical_name: mod for canonical_name, mod in self.items() if "." not in canonical_name})
    return ns


@contextlib.contextmanager
def zero_framework_sandbox(
  source_fw: str,
  target_fw: str,
  semantics: SemanticsManager | None = None,
) -> typing.Iterator[SandboxNamespace]:
  """Build an isolated ``SandboxNamespace`` binding framework module paths to ``zero-*`` modules.

  Args:
      source_fw: Source framework identifier.
      target_fw: Target framework identifier.
      semantics: Optional ``SemanticsManager`` instance.

  Yields:
      SandboxNamespace: Isolated mapping from canonical module names to imported ``zero-*`` modules.
  """
  ensure_compiler_and_zero_on_path()
  src_traits = _resolve_traits(source_fw, semantics)
  tgt_traits = _resolve_traits(target_fw, semantics)

  alias_map: dict[str, str] = dict(_SUPPORT_ZERO_ALIASES)
  alias_map.update(src_traits.zero_module_aliases)
  alias_map.update(tgt_traits.zero_module_aliases)

  active_bindings = SandboxNamespace()
  for canonical_name, zero_name in alias_map.items():
    try:
      zero_mod = importlib.import_module(zero_name)
    except (ImportError, SyntaxError, OSError, ValueError, TypeError, AttributeError, RuntimeError) as exc:
      logging.getLogger(__name__).debug("Zero runtime module '%s' unavailable: %s", zero_name, exc)
      continue
    active_bindings[canonical_name] = zero_mod

  yield active_bindings


class _TracedLayerInstance:
  """Runtime layer instance that records lazy input dimensions and parameter tensors."""

  def __init__(
    self,
    spec: DeclaredLayerSpec,
    seed: int,
    resolved_constructors: dict[str, ConstructorDimResolution],
    parameters: dict[str, OnDeviceTraceTensor],
  ) -> None:
    """Initialize the traced layer wrapper.

    Args:
        spec: Declared layer specification from ``__init__``/``setup``.
        seed: Random seed for parameter initialization.
        resolved_constructors: Mutable dictionary populated on first forward call.
        parameters: Mutable dictionary of model parameter tensors.
    """
    self.spec = spec
    self.seed = seed
    self._resolved_constructors = resolved_constructors
    self._parameters = parameters

  def _ensure_params(self, w_key: str, b_key: str, w_shape: tuple[int, ...], out_dim: int, w_scale: float) -> None:
    """Lazily initialize seeded weight and bias tensors for ``w_key`` and ``b_key``.

    Args:
        w_key: Parameter weight dictionary key.
        b_key: Parameter bias dictionary key.
        w_shape: Weight tensor shape tuple.
        out_dim: Bias vector dimension.
        w_scale: Random initialization scale for weights.
    """
    if w_key in self._parameters:
      return
    self._parameters[w_key] = OnDeviceTraceTensor(
      shape=w_shape,
      data=_generate_seeded_values(_prod(w_shape), self.seed + len(self._parameters) + 1, "float32", w_scale),
    )
    self._parameters[b_key] = OnDeviceTraceTensor(
      shape=(out_dim,),
      data=_generate_seeded_values(out_dim, self.seed + len(self._parameters) + 2, "float32", 0.1),
    )

  def __call__(self, x: OnDeviceTraceTensor) -> OnDeviceTraceTensor:
    """Execute the layer forward pass, lazily materializing parameters on first call.

    Args:
        x: Input ``OnDeviceTraceTensor``.

    Returns:
        Output ``OnDeviceTraceTensor``.
    """
    op = self.spec.op_type
    layer_name = self.spec.layer_name

    activation_ops: dict[str, typing.Callable[[OnDeviceTraceTensor], OnDeviceTraceTensor]] = {
      "Flatten": lambda t: t.flatten(1),
      "ReLU": lambda t: t.relu(),
      "relu": lambda t: t.relu(),
      "Sigmoid": lambda t: t.sigmoid(),
      "sigmoid": lambda t: t.sigmoid(),
      "Tanh": lambda t: t.tanh(),
      "tanh": lambda t: t.tanh(),
    }
    if op in activation_ops:
      return activation_ops[op](x)

    in_dim = x.shape[1] if (op.startswith(("Conv", "LazyConv"))) and len(x.shape) > 1 else (x.shape[-1] if x.shape else 4)
    out_dim = self.spec.out_features if self.spec.out_features is not None else in_dim

    if self.spec.is_lazy or self.spec.in_features is None or self.spec.in_features != in_dim:
      self._resolved_constructors[layer_name] = ConstructorDimResolution(
        layer_name=layer_name,
        op_type=op,
        inferred_in_features=in_dim,
        out_features=out_dim,
      )

    w_key = f"{layer_name}.weight"
    b_key = f"{layer_name}.bias"
    if op in ("Conv1d", "Conv1D", "Conv2d", "Conv2D", "Conv3d", "Conv3D", "Conv", "LazyConv2d"):
      w_shape: tuple[int, ...] = (out_dim, in_dim, 3, 3) if ("2" in op or op == "Conv") else (out_dim, in_dim, 3)
      self._ensure_params(w_key, b_key, w_shape, out_dim, 0.25)
      return _conv_nd(x, self._parameters[w_key], self._parameters[b_key], out_dim)

    self._ensure_params(w_key, b_key, (out_dim, in_dim), out_dim, 0.3)
    return _matmul_2d(x, self._parameters[w_key], self._parameters[b_key])


__all__ = [
  "_SUPPORT_ZERO_ALIASES",
  "SandboxNamespace",
  "_StructuralTraitsAdapterProtocol",
  "_TracedLayerInstance",
  "_lookup_adapter",
  "_resolve_traits",
  "zero_framework_sandbox",
]
