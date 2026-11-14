"""Runtime helpers for generated verification tests.

This module contains the reference implementation of ``verify_results``,
a robust recursive comparison utility for validating equivalence between
arbitrary data structures (Arrays, Lists, Dicts) across different frameworks.

It also defines the ``ensure_determinism`` fixture, which is injected into
generated tests to enforce reproducibility by seeding RNGs for Torch, JAX,
TensorFlow, NumPy, and Python.
"""

import importlib.util
import logging
import random
import sys
import types
import typing

import numpy as np
import pytest

from ml_switcheroo.enums import RuntimeTensorOrValue


@typing.runtime_checkable
class _CudaRngProtocol(typing.Protocol):
  """Protocol for CUDA device RNG controllers exposing ``is_available`` and ``manual_seed_all``."""

  def is_available(self) -> bool:
    """Return whether CUDA devices are available.

    Returns:
        bool: ``True`` if CUDA is available.
    """
    ...

  def manual_seed_all(self, seed: int) -> RuntimeTensorOrValue:
    """Seed all CUDA devices.

    Args:
        seed: Integer PRNG seed.

    Returns:
        Optional generator result.
    """
    ...


@typing.runtime_checkable
class _ManualSeedModuleProtocol(typing.Protocol):
  """Protocol for modules exposing ``manual_seed`` (and optional ``cuda`` attribute)."""

  def manual_seed(self, seed: int) -> RuntimeTensorOrValue:
    """Seed the host PRNG.

    Args:
        seed: Integer PRNG seed.

    Returns:
        Optional generator result.
    """
    ...


@typing.runtime_checkable
class _CudaContainerProtocol(typing.Protocol):
  """Protocol for modules exposing a ``cuda`` sub-namespace.

  Attributes:
      cuda: CUDA device sub-module.
  """

  cuda: _CudaRngProtocol


@typing.runtime_checkable
class _SetSeedNamespaceProtocol(typing.Protocol):
  """Protocol for RNG namespaces exposing ``set_seed``."""

  def set_seed(self, seed: int) -> None:
    """Set deterministic PRNG seed.

    Args:
        seed: Integer PRNG seed.
    """
    ...


@typing.runtime_checkable
class _SeedNamespaceProtocol(typing.Protocol):
  """Protocol for RNG namespaces exposing ``seed``."""

  def seed(self, seed: int) -> None:
    """Set deterministic PRNG seed.

    Args:
        seed: Integer PRNG seed.
    """
    ...


@typing.runtime_checkable
class _RandomAttrProtocol(typing.Protocol):
  """Protocol for modules exposing a ``random`` RNG sub-module.

  Attributes:
      random: Random number generation sub-module.
  """

  random: _SetSeedNamespaceProtocol | _SeedNamespaceProtocol


@typing.runtime_checkable
class _CoreAttrProtocol(typing.Protocol):
  """Protocol for root packages exposing a ``core`` sub-module.

  Attributes:
      core: Core array sub-module.
  """

  core: _RandomAttrProtocol


@typing.runtime_checkable
class _UtilsSetRandomSeedProtocol(typing.Protocol):
  """Protocol for ``utils`` namespaces exposing ``set_random_seed``."""

  def set_random_seed(self, seed: int) -> None:
    """Set framework-wide random seed.

    Args:
        seed: Integer PRNG seed.
    """
    ...


@typing.runtime_checkable
class _UtilsAttrProtocol(typing.Protocol):
  """Protocol for modules exposing a ``utils`` sub-module.

  Attributes:
      utils: Utilities sub-module.
  """

  utils: _UtilsSetRandomSeedProtocol


_SeedTargetModule: typing.TypeAlias = "types.ModuleType | _RandomAttrProtocol"


def _resolve_loaded_framework_module(pkg_path: str) -> "types.ModuleType | _RandomAttrProtocol | None":
  """Locate an already-imported framework module in ``sys.modules`` without importing new packages.

  Args:
      pkg_path: Dotted package path declared by ``adapter.import_alias[0]``.

  Returns:
      Loaded module or ``.core`` namespace if present in ``sys.modules``, otherwise ``None``.
  """
  mod = sys.modules.get(pkg_path)
  if mod is not None:
    return mod
  root_pkg = pkg_path.split(".")[0]
  root_mod = sys.modules.get(root_pkg)
  if root_mod is None:
    return None
  return root_mod.core if (pkg_path.endswith(".core") and isinstance(root_mod, _CoreAttrProtocol)) else root_mod


def _apply_rng_seed_method(fw_key: str, mod: "_SeedTargetModule", method: str, seed: int) -> None:
  """Invoke a single configured RNG seed method on ``mod``.

  Args:
      fw_key: Framework identifier for debug diagnostics.
      mod: Loaded framework module or sub-namespace.
      method: Relative RNG method specification (e.g. ``'manual_seed'``, ``'random.seed'``).
      seed: Integer PRNG seed value.
  """
  try:
    if method == "manual_seed" and isinstance(mod, _ManualSeedModuleProtocol):
      mod.manual_seed(seed)
      if isinstance(mod, _CudaContainerProtocol) and isinstance(mod.cuda, _CudaRngProtocol) and mod.cuda.is_available():
        mod.cuda.manual_seed_all(seed)
    elif method in ("set_seed", "random.set_seed") and isinstance(mod, _RandomAttrProtocol):
      rnd = mod.random
      if isinstance(rnd, _SetSeedNamespaceProtocol):
        rnd.set_seed(seed)
    elif method == "random.seed" and isinstance(mod, _RandomAttrProtocol):
      rnd2 = mod.random
      if isinstance(rnd2, _SeedNamespaceProtocol):
        rnd2.seed(seed)
    elif method == "utils.set_random_seed" and isinstance(mod, _UtilsAttrProtocol):
      if isinstance(mod.utils, _UtilsSetRandomSeedProtocol):
        mod.utils.set_random_seed(seed)
  except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
    logging.getLogger(__name__).debug("RNG seeding (%s.%s) skipped: %s", fw_key, method, exc)


def seed_framework_rngs(seed: int = 42) -> None:
  """Seed all currently imported frameworks according to ``FrameworkAdapter.rng_seed_methods``.

  Args:
      seed: Integer PRNG seed value.
  """
  from ml_switcheroo.frameworks.base import available_frameworks, get_adapter

  for fw_key in available_frameworks():
    adapter = get_adapter(fw_key)
    if adapter is None:
      continue
    try:
      seed_methods = adapter.rng_seed_methods
      pkg_path = adapter.import_alias[0]
    except AttributeError:
      continue
    if not seed_methods:
      continue
    mod = _resolve_loaded_framework_module(pkg_path)
    if mod is None:
      continue
    for method in seed_methods:
      _apply_rng_seed_method(fw_key, mod, method, seed)


@pytest.fixture(autouse=True)
def ensure_determinism() -> None:
  """Auto-inject fixed seeds for reproducibility across Python, NumPy, and registered frameworks."""
  random.seed(42)
  np.random.seed(42)
  from ml_switcheroo.generated_tests.runtime import seed_framework_rngs

  seed_framework_rngs(42)


def _verify_with_chex(
  ref: RuntimeTensorOrValue,
  val: RuntimeTensorOrValue,
  rtol: float,
  atol: float,
  exact: bool,
) -> bool | None:
  """Attempt PyTree structural verification via ``chex.assert_trees_all_close`` when available.

  Args:
      ref: Reference value.
      val: Candidate value.
      rtol: Relative tolerance.
      atol: Absolute tolerance.
      exact: Whether to enforce zero tolerance.

  Returns:
      ``True`` or ``False`` if Chex definitively evaluated equivalence, or ``None`` to fall back.
  """
  raw_chex = sys.modules.get("chex")
  if raw_chex is None and importlib.util.find_spec("chex") is None:
    return None
  try:
    chex_mod = raw_chex if raw_chex is not None else importlib.import_module("chex")
    eff_rtol = 0.0 if exact else rtol
    eff_atol = 0.0 if exact else atol
    chex_mod.assert_trees_all_close(ref, val, rtol=eff_rtol, atol=eff_atol)
    return True
  except AssertionError:
    return False
  except (ImportError, AttributeError, TypeError, ValueError, RuntimeError):
    return None


def _verify_leaf_arrays(
  ref: RuntimeTensorOrValue,
  val: RuntimeTensorOrValue,
  rtol: float,
  atol: float,
  exact: bool,
) -> bool:
  """Compare leaf array or scalar values using NumPy tolerances or exact equality.

  Args:
      ref: Reference leaf value.
      val: Candidate leaf value.
      rtol: Relative tolerance for floating/complex values.
      atol: Absolute tolerance for floating/complex values.
      exact: Whether to disable fuzzy tolerance matching.

  Returns:
      ``True`` if leaf values are equivalent, otherwise ``False``.
  """
  try:
    np_ref = np.asanyarray(ref)
    np_val = np.asanyarray(val)
    if np_ref.shape != np_val.shape and not (np_ref.size == 1 and np_val.size == 1):
      return False
    if exact or np_ref.dtype.kind not in {"f", "c"}:
      return bool(np.array_equal(np_ref, np_val))
    return bool(np.allclose(np_ref, np_val, rtol=rtol, atol=atol, equal_nan=True))
  except (TypeError, ValueError, RuntimeError):
    return bool(ref == val)


def verify_results(
  ref: RuntimeTensorOrValue,
  val: RuntimeTensorOrValue,
  rtol: float = 1e-3,
  atol: float = 1e-4,
  exact: bool = False,
) -> bool:
  """Cross-framework comparison helper.

  Recursively compares data structures (Lists, Dicts, Tuples, Arrays).

  Modes:
  - **Fuzzy (Default)**: Uses `np.allclose` with tolerances for floats.
  - **Exact**: Enforces strict equality (ids for None, `np.array_equal` for arrays).

  Args:
      ref (int | float | str | bool | list | dict | tuple | np.ndarray): The reference value (e.g. from Source Framework).
      val (int | float | str | bool | list | dict | tuple | np.ndarray): The candidate value (e.g. from Target Framework).
      rtol (float): Relative tolerance for floating point comparison.
      atol (float): Absolute tolerance for floating point comparison.
      exact (bool): If True, disables fuzzy matching.

  Returns:
      bool: True if values are considered equivalent.

  """
  if ref is None or val is None:
    return ref is val

  chex_res = _verify_with_chex(ref, val, rtol, atol, exact)
  if chex_res is not None:
    return chex_res

  if isinstance(ref, dict) and isinstance(val, dict):
    return ref.keys() == val.keys() and all(verify_results(ref[k], val[k], rtol, atol, exact=exact) for k in ref)

  if isinstance(ref, (list, tuple)) and isinstance(val, (list, tuple)):
    return len(ref) == len(val) and all(
      verify_results(r, v, rtol, atol, exact=exact) for r, v in zip(ref, val, strict=False)
    )

  return _verify_leaf_arrays(ref, val, rtol, atol, exact)
