"""Tests for Numpy framework adapter."""

import typing

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
          Computed test result or fixture value.
      """
      return CpuDummy()

  class CpuDummy:
    """Test helper class CpuDummy."""

    def cpu(self) -> "NumpyDummy":
      """Execute cpu helper logic.

      Returns:
          Computed test result or fixture value.
      """
      return NumpyDummy()

  class NumpyDummy:
    """Test helper class NumpyDummy."""

    def numpy(self) -> str:
      """Execute numpy helper logic.

      Returns:
          Computed test result or fixture value.
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
          Computed test result or fixture value.
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
      self, dtype: typing.Optional[str] = None, copy: typing.Optional[bool] = None
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.int64]]:
      """Execute array helper logic.

      Args:
          dtype: Test argument or fixture.
          copy: Test argument or fixture.

      Returns:
          Computed test result or fixture value.
      """
      return np.array([1], dtype=dtype)

  assert np.array_equal(adapter.convert(ArrayDummy()), np.array([1]))  # type: ignore[arg-type]

  class ArrayFail:
    """Test helper class ArrayFail."""

    def __array__(self, dtype: typing.Optional[str] = None, copy: typing.Optional[bool] = None) -> None:
      """Execute array helper logic.

      Args:
          dtype: Test argument or fixture.
          copy: Test argument or fixture.
      """
      raise RuntimeError("fail")

  adapter.convert(ArrayFail())
