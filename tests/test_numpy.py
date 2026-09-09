"""Tests for Numpy framework adapter."""

from unittest.mock import MagicMock, patch

from ml_switcheroo.frameworks.numpy import NumpyAdapter


def test_numpy_adapter_properties() -> None:
  """Docstring."""
  adapter: NumpyAdapter = NumpyAdapter()

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

  orig_import = __import__

  def mock_import_no_np(name, *args, **kwargs):
    """Docstring."""
    if name == "numpy":
      raise ImportError("mock no numpy")
    return orig_import(name, *args, **kwargs)

  with patch("builtins.__import__", side_effect=mock_import_no_np):
    adapter.convert([1, 2])

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
  adapter.get_tiered_examples()


def test_numpy_convert_branches() -> None:
  """Docstring."""
  adapter: NumpyAdapter = NumpyAdapter()
  assert adapter.convert({"a": 1}) == {"a": 1}

  class DetachDummy:
    """Docstring."""

    def detach(self) -> object:
      """Docstring."""

      class CpuDummy:
        """Docstring."""

        def cpu(self) -> object:
          """Docstring."""

          class NumpyDummy:
            """Docstring."""

            def numpy(self) -> str:
              """Docstring."""
              return "detached"

          return NumpyDummy()

      return CpuDummy()

  assert adapter.convert(DetachDummy()) == "detached"

  class DetachFail:
    """Docstring."""

    def detach(self) -> None:
      """Docstring."""
      raise Exception("fail")

  adapter.convert(DetachFail())

  class NumpyDummy2:
    """Docstring."""

    def numpy(self) -> str:
      """Docstring."""
      return "numpy"

  assert adapter.convert(NumpyDummy2()) == "numpy"

  class NumpyFail:
    """Docstring."""

    def numpy(self) -> None:
      """Docstring."""
      raise Exception("fail")

  adapter.convert(NumpyFail())

  class ArrayDummy:
    """Docstring."""

    def __array__(self, dtype: object = None) -> object:
      """Docstring."""
      import numpy as np

      return np.array([1], dtype=dtype)

  import numpy as np

  assert np.array_equal(adapter.convert(ArrayDummy()), np.array([1]))

  class ArrayFail:
    """Docstring."""

    def __array__(self) -> None:
      """Docstring."""
      raise Exception("fail")

  adapter.convert(ArrayFail())
