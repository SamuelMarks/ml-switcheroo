"""Test suite for the Harness Generator Missing module."""

from typing import Dict, Any, Tuple


def test_harness_generate_template() -> None:
  """Verifies the behavior of harness generate template."""
  from ml_switcheroo.testing.harness_generator import HarnessGenerator
  from pathlib import Path

  hg: HarnessGenerator = HarnessGenerator()
  semantics: Dict[str, Any] = {
    "op1": {"std_args": [["arg1", "int"], {"name": "arg2", "type": "float"}, {"name": "arg3"}]}
  }
  source: Path = Path("source.py")
  target: Path = Path("target.py")
  out: Path = Path("out.py")
  hg.generate(source, target, out, "jax", "torch", semantics)


def test_harness_adapter_shim_exceptions() -> None:
  """Verifies the behavior of harness adapter shim exceptions."""
  from ml_switcheroo.testing.harness_generator import HarnessGenerator

  hg: HarnessGenerator = HarnessGenerator()
  with __import__("unittest.mock").mock.patch("ml_switcheroo.testing.harness_generator.get_adapter", return_value=None):
    res: Tuple[str, str, str] = hg._build_dynamic_init("fake_fw")
    assert res == ("", "", "pass")

  class MockAdapter:
    """Mock Adapter class for testing purposes."""

    def get_to_numpy_code(self) -> str:
      """Mock implementation of get to NumPy code."""
      raise Exception("Fail")

  with __import__("unittest.mock").mock.patch(
    "ml_switcheroo.testing.harness_generator.get_adapter", return_value=MockAdapter()
  ):
    res2: str = hg._build_result_normalization("jax", "torch")
    assert res2 == ""


def test_harness_extractor_oserror() -> None:
  """Verifies the behavior of harness extractor oserror."""
  from ml_switcheroo.testing.harness_generator import HarnessGenerator

  hg: HarnessGenerator = HarnessGenerator()
  with __import__("unittest.mock").mock.patch(
    "ml_switcheroo.utils.code_extractor.CodeExtractor.extract_class", side_effect=OSError("fail")
  ):
    try:
      hg._bundle_fuzzer_dependencies()
    except OSError:
      pass


def test_harness_extract_module_functions_oserror() -> None:
  """Verifies the behavior of harness extract module functions oserror."""
  from ml_switcheroo.testing.harness_generator import HarnessGenerator
  import inspect

  hg: HarnessGenerator = HarnessGenerator()
  original_getsource: Any = inspect.getsource

  def mock_getsource(obj: Any) -> str:
    """Provides a mock getsource for testing."""
    if inspect.isfunction(obj):
      raise OSError("fail")
    return original_getsource(obj)

  with __import__("unittest.mock").mock.patch("inspect.getsource", side_effect=mock_getsource):
    hg._bundle_fuzzer_dependencies()


def test_harness_build_result_normalization_flax() -> None:
  """Verifies the behavior of harness build result normalization Flax."""
  from ml_switcheroo.testing.harness_generator import HarnessGenerator

  hg: HarnessGenerator = HarnessGenerator()
  res: str = hg._build_result_normalization("flax_nnx", "torch")
  assert "jax" in res or "flax_nnx" in res


def test_harness_generate_adapter_shim_oserror() -> None:
  """Verifies the behavior of harness generate adapter shim oserror."""
  from ml_switcheroo.testing.harness_generator import HarnessGenerator
  import inspect

  hg: HarnessGenerator = HarnessGenerator()
  original_getsource: Any = inspect.getsource

  def mock_getsource(obj: Any) -> str:
    """Provides a mock getsource for testing."""
    if hasattr(obj, "__name__") and getattr(obj, "__name__") == "convert":
      raise OSError("fail")
    return original_getsource(obj)

  with __import__("unittest.mock").mock.patch("inspect.getsource", side_effect=mock_getsource):
    hg._generate_adapter_shim()


def test_harness_generate_adapter_shim_no_convert() -> None:
  """Verifies the behavior of harness generate adapter shim no convert."""
  from ml_switcheroo.testing.harness_generator import HarnessGenerator
  from ml_switcheroo.frameworks.base import _ADAPTER_REGISTRY

  class NoConvertAdapter:
    """Test suite for the No Convert Adapter component."""

    pass

  _ADAPTER_REGISTRY["fake_fw"] = NoConvertAdapter
  try:
    hg: HarnessGenerator = HarnessGenerator()
    hg._generate_adapter_shim()
  finally:
    del _ADAPTER_REGISTRY["fake_fw"]


def test_harness_build_dynamic_init_with_magic_args() -> None:
  """Test method."""
  from ml_switcheroo.testing.harness_generator import HarnessGenerator
  import unittest.mock as mock

  class MockAdapter:
    """Test class."""

    harness_imports: list[str] = ["import something"]
    declared_magic_args: list[str] = ["my_magic_arg"]

    def get_harness_init_code(self) -> str:
      """Get harness init code."""
      return "def my_helper(): pass"

  hg: HarnessGenerator = HarnessGenerator()
  with mock.patch("ml_switcheroo.testing.harness_generator.get_adapter", return_value=MockAdapter()):
    with mock.patch(
      "ml_switcheroo.testing.signature_extractor.SignatureExtractor.extract_first_function_name", return_value="my_helper"
    ):
      imports_str: str
      init_code: str
      final_logic: str
      imports_str, init_code, final_logic = hg._build_dynamic_init("mock_fw")
      assert "import something" in imports_str
      assert "my_helper" in init_code
      assert "tgt_inputs[tp] = val" in final_logic
      assert "val = None" in final_logic


def test_harness_injector_leave_module() -> None:
  """Test method."""
  import libcst as cst
  from ml_switcheroo.testing.harness_generator import HarnessInjector

  injector: HarnessInjector = HarnessInjector(
    imports_block="import sys\n",
    init_helpers_block="def helper(): pass\n",
    fuzzer_block="",
    to_numpy_block="",
    injection_block="",
    source_fw="jax",
    target_fw="torch",
    source_path="in.py",
    target_path="out.py",
    hints_json="{}",
  )
  module_code: str = "def to_numpy(): pass\n"
  module: cst.Module = cst.parse_module(module_code)
  updated_module: cst.Module = module.visit(injector)
  updated_code: str = updated_module.code
  assert "import sys" in updated_code
  assert "def helper" in updated_code


def test_harness_injector_leave_functiondef_no_block() -> None:
  """Test method."""
  import libcst as cst
  from ml_switcheroo.testing.harness_generator import HarnessInjector

  injector: HarnessInjector = HarnessInjector(
    imports_block="",
    init_helpers_block="",
    fuzzer_block="",
    to_numpy_block="   ",
    injection_block="",
    source_fw="jax",
    target_fw="torch",
    source_path="in.py",
    target_path="out.py",
    hints_json="{}",
  )
  module_code: str = "def to_numpy(): pass\n"
  module: cst.Module = cst.parse_module(module_code)
  updated_module: cst.Module = module.visit(injector)
  assert updated_module.code == module.code


def test_harness_injector_leave_if_with_injection() -> None:
  """Test method."""
  import libcst as cst
  from ml_switcheroo.testing.harness_generator import HarnessInjector

  injector: HarnessInjector = HarnessInjector(
    imports_block="",
    init_helpers_block="",
    fuzzer_block="",
    to_numpy_block="",
    injection_block="tgt_inputs[tp] = 42\n",
    source_fw="jax",
    target_fw="torch",
    source_path="in.py",
    target_path="out.py",
    hints_json="{}",
  )
  module_code: str = "if tp not in tgt_inputs:\n    pass\n"
  module: cst.Module = cst.parse_module(module_code)
  updated_module: cst.Module = module.visit(injector)
  assert "tgt_inputs[tp] = 42" in updated_module.code
