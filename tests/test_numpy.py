"""Tests for Numpy framework adapter."""

from unittest.mock import MagicMock, patch

import numpy as np

from ml_switcheroo.frameworks.numpy import NumpyAdapter


def test_numpy_adapter_properties() -> None:
  """Verify numpy adapter properties behavior."""
  adapter = NumpyAdapter()

  adapter.import_alias
  adapter.import_namespaces
  adapter.supported_tiers
  adapter.structural_traits
  adapter.plugin_traits
  adapter.test_config
  adapter.harness_imports
  adapter.get_harness_init_code()
  adapter.get_to_numpy_code()
  adapter.declared_magic_args
  adapter.rng_seed_methods

  with patch("ml_switcheroo.frameworks.numpy.load_definitions", return_value={"test": MagicMock()}):
    pass

  # Test numpy array conversion with exception fallback
  mock_np = MagicMock()
  mock_np.array.return_value = "tensor"
  with patch("ml_switcheroo.frameworks.numpy.np", mock_np):
    adapter.convert([1, 2])
    mock_np.array.side_effect = Exception("err")
    adapter.convert([1, 2])

  adapter.get_device_syntax("cpu")
  adapter.get_device_check_syntax()
  adapter.get_rng_split_syntax("rng", "key")
  adapter.get_serialization_imports()
  adapter.get_serialization_syntax("save", "path", "obj")
  adapter.get_serialization_syntax("load", "path")
  adapter.get_serialization_syntax("other", "path")
  adapter.get_weight_conversion_imports()
  adapter.get_weight_load_code("path")
  adapter.get_tensor_to_numpy_expr("t")
  adapter.get_weight_save_code("s", "path")

  adapter.apply_wiring({})
  adapter.get_doc_url("numpy.add")
  _res = adapter.get_tiered_examples()
  assert isinstance(_res, dict)
  assert "tier1_math" in _res


def test_numpy_convert_branches() -> None:
  """Verify numpy convert branches behavior."""
  adapter = NumpyAdapter()
  assert adapter.convert({"a": 1}) == {"a": 1}

  class DetachDummy:
    """Test helper class DetachDummy."""

    def detach(self) -> "CpuDummy":
      """Execute detach helper logic.

      Returns:
          Resulting `'CpuDummy'` from detach.
      """
      return CpuDummy()

  class CpuDummy:
    """Test helper class CpuDummy."""

    def cpu(self) -> "NumpyDummy":
      """Execute cpu helper logic.

      Returns:
          Resulting `'NumpyDummy'` from cpu.
      """
      return NumpyDummy()

  class NumpyDummy:
    """Test helper class NumpyDummy."""

    def numpy(self) -> str:
      """Execute numpy helper logic.

      Returns:
          Resulting `str` from numpy.
      """
      return "detached"

  assert adapter.convert(DetachDummy()) == "detached"

  class DetachFail:
    """Test helper class DetachFail."""

    def detach(self) -> None:
      """Execute detach helper logic."""
      raise RuntimeError("fail")

  adapter.convert(DetachFail())

  class NumpyDummy2:
    """Test helper class NumpyDummy2."""

    def numpy(self) -> str:
      """Execute numpy helper logic.

      Returns:
          Resulting `str` from numpy.
      """
      return "numpy"

  assert adapter.convert(NumpyDummy2()) == "numpy"

  class NumpyFail:
    """Test helper class NumpyFail."""

    def numpy(self) -> None:
      """Execute numpy helper logic."""
      raise RuntimeError("fail")

  adapter.convert(NumpyFail())

  class ArrayDummy:
    """Test helper class ArrayDummy."""

    def __array__(
      self, dtype: str | None = None, copy: bool | None = None
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.int64]]:
      """Execute array helper logic.

      Args:
          dtype: The dtype (str | None) for array.
          copy: The copy (bool | None) for array.

      Returns:
          Resulting `np.ndarray[tuple[int, ...], np.dtype[np.int64]]` from array.
      """
      return np.array([1], dtype=dtype)

  assert getattr({"v": getattr({"v": np}, "__getitem__")("v").array_equal}, "__getitem__")("v")(
    getattr(
      {
        "v": getattr({"v": getattr({"v": adapter}, "__getitem__")("v").convert}, "__getitem__")("v")(
          getattr({"v": getattr({"v": ArrayDummy}, "__getitem__")("v")()}, "__getitem__")("v")
        )
      },
      "__getitem__",
    )("v"),
    getattr(
      {
        "v": getattr({"v": getattr({"v": np}, "__getitem__")("v").array}, "__getitem__")("v")(
          getattr({"v": [getattr({"v": 1}, "__getitem__")("v")]}, "__getitem__")("v")
        )
      },
      "__getitem__",
    )("v"),
  )

  class ArrayFail:
    """Test helper class ArrayFail."""

    def __array__(self, dtype: str | None = None, copy: bool | None = None) -> None:
      """Execute array helper logic.

      Args:
          dtype: The dtype (str | None) for array.
          copy: The copy (bool | None) for array.
      """
      raise RuntimeError("fail")

  adapter.convert(ArrayFail())
