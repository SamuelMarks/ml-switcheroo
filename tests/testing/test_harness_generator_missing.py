"""Test suite for the Harness Generator Missing module."""


def test_harness_generate_template():
  """Verifies the behavior of harness generate template."""
  from ml_switcheroo.testing.harness_generator import HarnessGenerator
  from pathlib import Path

  hg = HarnessGenerator()
  semantics = {"op1": {"std_args": [["arg1", "int"], {"name": "arg2", "type": "float"}, {"name": "arg3"}]}}
  source = Path("source.py")
  target = Path("target.py")
  out = Path("out.py")
  hg.generate(source, target, out, "jax", "torch", semantics)


def test_harness_adapter_shim_exceptions():
  """Verifies the behavior of harness adapter shim exceptions."""
  from ml_switcheroo.testing.harness_generator import HarnessGenerator

  hg = HarnessGenerator()
  with __import__("unittest.mock").mock.patch("ml_switcheroo.testing.harness_generator.get_adapter", return_value=None):
    res = hg._build_dynamic_init("fake_fw")
    assert res == ("", "", "pass")

  class MockAdapter:
    """Mock Adapter class for testing purposes."""

    def get_to_numpy_code(self):
      """Mock implementation of get to NumPy code."""
      raise Exception("Fail")

  with __import__("unittest.mock").mock.patch(
    "ml_switcheroo.testing.harness_generator.get_adapter", return_value=MockAdapter()
  ):
    res = hg._build_result_normalization("jax", "torch")
    assert res == ""


def test_harness_extractor_oserror():
  """Verifies the behavior of harness extractor oserror."""
  from ml_switcheroo.testing.harness_generator import HarnessGenerator

  hg = HarnessGenerator()
  with __import__("unittest.mock").mock.patch(
    "ml_switcheroo.utils.code_extractor.CodeExtractor.extract_class", side_effect=OSError("fail")
  ):
    try:
      hg._bundle_fuzzer_dependencies()
    except OSError:
      pass


def test_harness_extract_module_functions_oserror():
  """Verifies the behavior of harness extract module functions oserror."""
  from ml_switcheroo.testing.harness_generator import HarnessGenerator
  import inspect

  hg = HarnessGenerator()
  original_getsource = inspect.getsource

  def mock_getsource(obj):
    """Provides a mock getsource for testing."""
    if inspect.isfunction(obj):
      raise OSError("fail")
    return original_getsource(obj)

  with __import__("unittest.mock").mock.patch("inspect.getsource", side_effect=mock_getsource):
    hg._bundle_fuzzer_dependencies()


def test_harness_build_result_normalization_flax():
  """Verifies the behavior of harness build result normalization Flax."""
  from ml_switcheroo.testing.harness_generator import HarnessGenerator

  hg = HarnessGenerator()
  res = hg._build_result_normalization("flax_nnx", "torch")
  assert "jax" in res or "flax_nnx" in res


def test_harness_generate_adapter_shim_oserror():
  """Verifies the behavior of harness generate adapter shim oserror."""
  from ml_switcheroo.testing.harness_generator import HarnessGenerator
  import inspect

  hg = HarnessGenerator()
  original_getsource = inspect.getsource

  def mock_getsource(obj):
    """Provides a mock getsource for testing."""
    if hasattr(obj, "__name__") and obj.__name__ == "convert":
      raise OSError("fail")
    return original_getsource(obj)

  with __import__("unittest.mock").mock.patch("inspect.getsource", side_effect=mock_getsource):
    hg._generate_adapter_shim()


def test_harness_generate_adapter_shim_no_convert():
  """Verifies the behavior of harness generate adapter shim no convert."""
  from ml_switcheroo.testing.harness_generator import HarnessGenerator
  from ml_switcheroo.frameworks.base import _ADAPTER_REGISTRY

  class NoConvertAdapter:
    """Test suite for the No Convert Adapter component."""

    pass

  _ADAPTER_REGISTRY["fake_fw"] = NoConvertAdapter
  try:
    hg = HarnessGenerator()
    hg._generate_adapter_shim()
  finally:
    del _ADAPTER_REGISTRY["fake_fw"]
