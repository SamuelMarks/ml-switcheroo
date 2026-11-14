"""Unit tests for `NumpyAdapter` (`ml_switcheroo.frameworks.numpy`)."""

import importlib
import sys
import typing

import numpy as np
import pytest
from ml_switcheroo_ir.schema.ghost import SemanticTier
from pydantic import JsonValue

import ml_switcheroo.frameworks.numpy as np_mod
from ml_switcheroo.frameworks.numpy import NumpyAdapter


def test_numpy_adapter_metadata_traits_and_syntax() -> None:
  """Verify `NumpyAdapter` metadata, namespaces, traits, device/serialization/weight syntax, and docs."""
  adapter = NumpyAdapter()
  assert adapter.display_name == "numpy"
  assert adapter.import_alias == ("numpy", "np")
  assert "numpy" in adapter.import_namespaces
  assert "import numpy as np" in adapter.test_config["import"]
  assert adapter.harness_imports == []
  assert adapter.get_harness_init_code() == ""
  assert "isinstance(obj, np.ndarray)" in adapter.get_to_numpy_code()
  assert SemanticTier.ARRAY_API in adapter.supported_tiers
  assert adapter.declared_magic_args == []
  assert adapter.structural_traits.auto_strip_magic_args is True
  assert adapter.plugin_traits.has_numpy_compatible_arrays is True
  assert isinstance(adapter.definitions, dict)
  assert "seed" in adapter.rng_seed_methods

  assert adapter.get_device_syntax("cuda") == "'cpu'"
  assert adapter.get_device_syntax("cpu") == "'cpu'"
  assert adapter.get_device_check_syntax() == "False"
  assert adapter.get_rng_split_syntax("rng", "key") == "pass"

  assert adapter.get_serialization_imports() == ["import numpy as np"]
  assert adapter.get_serialization_syntax("save", "file", "obj") == "np.save(file=file, arr=obj)"
  assert adapter.get_serialization_syntax("load", "file") == "np.load(file=file)"
  assert adapter.get_serialization_syntax("invalid", "file") == ""
  assert adapter.get_serialization_syntax("save", "file", None) == ""

  assert adapter.get_weight_conversion_imports() == ["import numpy as np"]
  assert "loaded = np.load(path, allow_pickle=True)" in adapter.get_weight_load_code("path")
  assert adapter.get_tensor_to_numpy_expr("t") == "t"
  assert adapter.get_weight_save_code("state", "path") == "np.savez_compressed(path, **state)"

  assert adapter.get_doc_url("numpy.array") == "https://numpy.org/doc/stable/reference/generated/numpy.array.html"
  examples = adapter.get_tiered_examples()
  assert {"tier1_math", "tier2_neural", "tier3_extras"} <= set(examples.keys())

  snapshot: dict[str, dict[str, JsonValue]] = {}
  adapter.apply_wiring(snapshot)
  assert snapshot == {}


def test_numpy_adapter_convert_protocols_and_missing_numpy_reload(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `NumpyAdapter.convert` across Torch/TF/`__array__` protocols, error fallbacks, and `np=None`.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  adapter = NumpyAdapter()
  assert adapter.convert([1, 2, 3]) == [1, 2, 3]
  assert adapter.convert({"a": 1}) == {"a": 1}
  raw_nd = np.array([1, 2])
  assert adapter.convert(raw_nd) is raw_nd

  class _DetachTensor:
    """Stub tensor implementing `.detach().cpu().numpy()`."""

    def detach(self) -> "_DetachTensor":
      """Return self.

      Returns:
          `_DetachTensor` instance.
      """
      return self

    def cpu(self) -> "_DetachTensor":
      """Return self.

      Returns:
          `_DetachTensor` instance.
      """
      return self

    def numpy(self) -> str:
      """Return `'from_detach'`.

      Returns:
          Literal `'from_detach'`.
      """
      return "from_detach"

  class _DetachFail:
    """Stub tensor raising `RuntimeError` in `.detach()`."""

    def detach(self) -> typing.Never:
      """Raise `RuntimeError`.

      Returns:
          Never returns; raises `RuntimeError`.

      Raises:
          RuntimeError: Always raised.
      """
      raise RuntimeError("detach fail")

  class _NumpyTensor:
    """Stub tensor implementing `.numpy()`."""

    def numpy(self) -> str:
      """Return `'from_numpy_method'`.

      Returns:
          Literal `'from_numpy_method'`.
      """
      return "from_numpy_method"

  class _NumpyFail:
    """Stub tensor raising `RuntimeError` in `.numpy()`."""

    def numpy(self) -> typing.Never:
      """Raise `RuntimeError`.

      Returns:
          Never returns; raises `RuntimeError`.

      Raises:
          RuntimeError: Always raised.
      """
      raise RuntimeError("numpy fail")

  class _ArrayObj:
    """Stub object implementing `__array__`."""

    def __array__(
      self, dtype: str | None = None, copy: bool | None = None
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.int64]]:
      """Return `np.array([7])`.

      Args:
          dtype: Optional dtype.
          copy: Optional copy flag.

      Returns:
          1D int64 NumPy array.
      """
      _ = copy
      return np.array([7], dtype=dtype)

  class _ArrayFail:
    """Stub object raising `RuntimeError` in `__array__`."""

    def __array__(self, dtype: str | None = None, copy: bool | None = None) -> typing.Never:
      """Raise `RuntimeError`.

      Args:
          dtype: Optional dtype.
          copy: Optional copy flag.

      Returns:
          Never returns; raises `RuntimeError`.

      Raises:
          RuntimeError: Always raised.
      """
      _ = (dtype, copy)
      raise RuntimeError("array fail")

  assert adapter.convert(_DetachTensor()) == "from_detach"
  df = _DetachFail()
  assert adapter.convert(df) is df
  assert adapter.convert(_NumpyTensor()) == "from_numpy_method"
  nf = _NumpyFail()
  assert adapter.convert(nf) is nf

  arr_conv = adapter.convert(_ArrayObj())
  assert isinstance(arr_conv, np.ndarray)
  assert np.array_equal(arr_conv, np.array([7]))

  af = _ArrayFail()
  assert adapter.convert(af) is af

  # When `np` is `None`, `__array__` object is returned unchanged
  monkeypatch.setattr(np_mod, "np", None)
  ao = _ArrayObj()
  assert adapter.convert(ao) is ao

  # Reload `ml_switcheroo.frameworks.numpy` with `numpy` blocked in `sys.modules`
  with pytest.MonkeyPatch.context() as mp:
    mp.setitem(sys.modules, "numpy", None)
    importlib.reload(np_mod)
    assert np_mod.np is None
  importlib.reload(np_mod)
  assert isinstance(NumpyAdapter().convert(_ArrayObj()), np.ndarray)
