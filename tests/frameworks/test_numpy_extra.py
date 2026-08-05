"""Test module."""

import sys


def test_numpy_init_missing(monkeypatch):
  """Test function."""
  import ml_switcheroo.frameworks.numpy as np_fw

  old_np = sys.modules.get("numpy")
  sys.modules["numpy"] = None
  import importlib

  try:
    importlib.reload(np_fw)
    assert getattr(np_fw, "np", None) is None
  finally:
    if old_np:
      sys.modules["numpy"] = old_np
    else:
      del sys.modules["numpy"]


def test_numpy_convert(monkeypatch):
  """Test function."""
  import ml_switcheroo.frameworks.numpy as np_fw
  import importlib

  importlib.reload(np_fw)

  mock_np = type("MockNP", (), {"array": lambda x: "mock_array"})
  monkeypatch.setattr(np_fw, "np", mock_np)

  adapter = np_fw.NumpyAdapter()

  assert adapter.convert([1, 2, 3]) == [1, 2, 3]
  assert adapter.convert({"a": 1}) == {"a": 1}

  class MockTorch:
    """A mock Torch tensor."""

    def detach(self):
      """Mocks detach."""
      return self

    def cpu(self):
      """Mocks cpu."""
      return self

    def numpy(self):
      """Mocks numpy."""
      return "numpy_tensor"

  assert adapter.convert(MockTorch()) == "numpy_tensor"

  class FailTorch:
    """A fake failing torch tensor."""

    def detach(self):
      """Mocks detach."""
      raise Exception("Fail")

  f = FailTorch()
  assert adapter.convert(f) is f

  class MockTF:
    """A mock TF tensor."""

    def numpy(self):
      """Mocks numpy."""
      return "tf_tensor"

  assert adapter.convert(MockTF()) == "tf_tensor"

  class FailTF:
    """A failing TF tensor."""

    def numpy(self):
      """Mocks numpy."""
      raise Exception("Fail")

  f2 = FailTF()
  assert adapter.convert(f2) is f2

  class MockArray:
    """A mock numpy array interface."""

    def __array__(self):
      """Gets array."""
      return []

  assert adapter.convert(MockArray()) == "mock_array"

  class FailArray:
    """A failing array."""

    def __array__(self):
      """Gets array."""
      return []  # trigger the lambda then raise

  def failing_array(x):
    """Mocks numpy array creation."""
    if isinstance(x, FailArray):
      raise Exception("Fail")
    return "mock_array2"

  mock_np2 = type("MockNP2", (), {"array": failing_array})
  monkeypatch.setattr(np_fw, "np", mock_np2)

  assert adapter.convert(MockArray()) == "mock_array2"
  f3 = FailArray()
  assert adapter.convert(f3) is f3


def test_numpy_properties():
  """Test function."""
  from ml_switcheroo.frameworks.numpy import NumpyAdapter

  adapter = NumpyAdapter()

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

  traits = adapter.plugin_traits
  assert traits.has_numpy_compatible_arrays is True


def test_numpy_doc_url():
  """Test function."""
  from ml_switcheroo.frameworks.numpy import NumpyAdapter

  adapter = NumpyAdapter()
  assert adapter.get_doc_url("numpy.array") == "https://numpy.org/doc/stable/reference/generated/numpy.array.html"


def test_numpy_test_config():
  """Test function."""
  from ml_switcheroo.frameworks.numpy import NumpyAdapter

  adapter = NumpyAdapter()
  assert "import numpy as np" in adapter.test_config["import"]


def test_numpy_get_to_numpy_code():
  """Test function."""
  from ml_switcheroo.frameworks.numpy import NumpyAdapter

  adapter = NumpyAdapter()
  assert "if isinstance(obj, np.ndarray)" in adapter.get_to_numpy_code()


def test_numpy_get_tiered_examples():
  """Test function."""
  from ml_switcheroo.frameworks.numpy import NumpyAdapter

  adapter = NumpyAdapter()
  assert "tier1_math" in adapter.get_tiered_examples()
  assert "tier2_neural" in adapter.get_tiered_examples()
  assert "tier3_extras" in adapter.get_tiered_examples()
