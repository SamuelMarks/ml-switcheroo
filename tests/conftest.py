"""Test suite for the Conftest module."""

import sys
import pytest
import warnings
from pathlib import Path
from typing import Callable, Optional, Generator
import typing
import importlib
from ml_switcheroo.core.rewriter import RewriterContext, RewriterPipeline, StructuralPass, ApiPass, AuxiliaryPass

warnings.filterwarnings("ignore", message=".*np\\.object.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="keras.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="tensorflow.*")
src_path: Path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))
try:
  importlib.import_module("ml_switcheroo.frameworks")
  from ml_switcheroo.frameworks.base import _ADAPTER_REGISTRY
except ImportError:
  _ADAPTER_REGISTRY = {}


class TestRewriter:
  """Test suite for the Rewriter component."""

  __test__ = False

  def __init__(self, semantics: typing.Any, config: typing.Any, symbol_table: typing.Optional[typing.Any] = None) -> None:
    """Initializes the TestRewriter instance."""
    self.context = RewriterContext(semantics, config, symbol_table)
    self.pipeline = RewriterPipeline([StructuralPass(), ApiPass(), AuxiliaryPass()])

  @property
  def ctx(self) -> typing.Any:
    """Helper to ctx."""
    return self.context.hook_context

  @property
  def semantics(self) -> typing.Any:
    """Helper to semantics."""
    return self.context.semantics

  def convert(self, module: typing.Any) -> typing.Any:
    """Converts ."""
    return self.pipeline.run(module, self.context)


class SnapshotAssert:
  """Test suite for the Snapshot Assert component."""

  def __init__(self, request: pytest.FixtureRequest) -> None:
    """Initializes the SnapshotAssert instance."""
    self.request = request
    self.test_name: str = request.node.name
    self.module_path: Path = Path(request.node.fspath).parent
    self.snapshot_dir: Path = self.module_path / "__snapshots__"
    self.update_mode: bool = request.config.getoption("--update-snapshots", default=False)

  def assert_match(self, content: str, extension: str = "txt", normalizer: Optional[Callable[[str], str]] = None) -> None:
    """Helper to assert match."""
    if not self.snapshot_dir.exists():
      self.snapshot_dir.mkdir(parents=True)
    snapshot_file: Path = self.snapshot_dir / f"{self.test_name}.{extension}"
    content = content.replace("\r\n", "\n")
    if self.update_mode or not snapshot_file.exists():
      normalized_to_write: str = normalizer(content) if normalizer else content
      snapshot_file.write_text(normalized_to_write, encoding="utf-8")
      if self.update_mode:
        return
    expected: str = snapshot_file.read_text(encoding="utf-8").replace("\r\n", "\n")
    lhs: str = content
    rhs: str = expected
    if normalizer:
      lhs = normalizer(lhs)
      rhs = normalizer(rhs)
    assert lhs == rhs, f"Snapshot mismatch for {snapshot_file.name}. Run with --update-snapshots to accept changes."


@pytest.fixture
def snapshot(request: pytest.FixtureRequest) -> SnapshotAssert:
  """Provides a mock snapshot for testing."""
  return SnapshotAssert(request)


@pytest.fixture(autouse=True)
def isolate_semantics_manager() -> Generator[None, None, None]:
  """Helper to isolate semantics manager."""
  from ml_switcheroo.semantics import manager as manager_module

  original_cache = manager_module._MANAGER_CACHE
  manager_module._MANAGER_CACHE = None
  yield
  manager_module._MANAGER_CACHE = original_cache


@pytest.fixture(autouse=True)
def isolate_hook_registry() -> Generator[None, None, None]:
  """Helper to isolate hook registry."""
  from ml_switcheroo.core.hooks import _HOOKS, _HOOK_METADATA
  import ml_switcheroo.core.hooks as hooks_module

  original_hooks = _HOOKS.copy()
  original_metadata = _HOOK_METADATA.copy()
  original_loaded = getattr(hooks_module, "_PLUGINS_LOADED", False)
  yield
  _HOOKS.clear()
  _HOOKS.update(original_hooks)
  _HOOK_METADATA.clear()
  _HOOK_METADATA.update(original_metadata)
  hooks_module._PLUGINS_LOADED = original_loaded


@pytest.fixture(autouse=True)
def isolate_framework_registry() -> Generator[None, None, None]:
  """Helper to isolate framework registry."""
  original_registry = _ADAPTER_REGISTRY.copy()
  yield
  _ADAPTER_REGISTRY.clear()
  _ADAPTER_REGISTRY.update(original_registry)


def pytest_addoption(parser: pytest.Parser) -> None:
  """Helper to pytest addoption."""
  parser.addoption("--update-snapshots", action="store_true", default=False, help="Update snapshots for visual tests")
