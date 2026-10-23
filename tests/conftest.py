"""Test suite for the Conftest module."""

import importlib
import sys
import typing
import warnings
from pathlib import Path

import libcst as cst
import pytest

from ml_switcheroo.analysis.symbol_table import SymbolTable
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.core.rewriter import ApiPass, AuxiliaryPass, RewriterContext, RewriterPipeline, StructuralPass
from ml_switcheroo.core.rewriter.context import SemanticsProtocol
from ml_switcheroo.semantics.manager import SemanticsManager

warnings.filterwarnings("ignore", message=".*np\\.object.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="keras.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="tensorflow.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow\\.lite.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="jax\\._src\\.deprecations.*")
src_path: Path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))
try:
  importlib.import_module("ml_switcheroo.frameworks")
  from ml_switcheroo.frameworks.base import _ADAPTER_REGISTRY
except ImportError:
  _ADAPTER_REGISTRY = {}


class TestRewriter:
  """Test helper class TestRewriter."""

  __test__ = False

  def __init__(
    self,
    semantics: SemanticsManager | SemanticsProtocol,
    config: RuntimeConfig | None = None,
    symbol_table: SymbolTable | None = None,
  ) -> None:
    """Initialize test helper instance.

    Args:
        semantics: SemanticsManager instance providing operation and trait definitions.
        config: RuntimeConfig specifying source and target frameworks.
        symbol_table: SymbolTable tracking variable types and aliases.
    """
    self.context = RewriterContext(semantics, config, symbol_table)
    self.pipeline = RewriterPipeline([StructuralPass(), ApiPass(), AuxiliaryPass()])

  @property
  def ctx(self) -> HookContext:
    """Helper to ctx.

    Returns:
        The `ctx` property value (HookContext).
    """
    return self.context.hook_context

  @property
  def semantics(self) -> SemanticsManager:
    """Helper to semantics.

    Returns:
        The `semantics` property value (SemanticsManager).
    """
    return self.context.semantics

  def convert(self, module: cst.Module) -> cst.Module:
    """Converts .

    Args:
        module: The module (cst.Module) for convert.

    Returns:
        Transpiled source code string or rewritten CST module.
    """
    return self.pipeline.run(module, self.context)


class SnapshotAssert:
  """Test helper class SnapshotAssert."""

  def __init__(self, request: pytest.FixtureRequest) -> None:
    """Initializes the SnapshotAssert instance.

    Args:
        request: Pytest fixture request context.
    """
    self.request = request
    self.test_name: str = request.node.name
    self.module_path: Path = Path(request.node.fspath).parent
    self.snapshot_dir: Path = self.module_path / "__snapshots__"
    self.update_mode: bool = request.config.getoption("--update-snapshots", default=False)

  def assert_match(
    self, content: str, extension: str = "txt", normalizer: typing.Callable[[str], str] | None = None
  ) -> None:
    """Helper to assert match.

    Args:
        content: The content (str) for assert match.
        extension: The extension (str) for assert match.
        normalizer: The normalizer (typing.Callable[[str], str] | None) for assert match.
    """
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
  """Execute snapshot helper logic.

  Args:
      request: Pytest fixture request context.

  Returns:
      Configured `snapshot` fixture value (SnapshotAssert).
  """
  return SnapshotAssert(request)


@pytest.fixture(autouse=True)
def isolate_semantics_manager() -> typing.Generator[None, None, None]:
  """Helper to isolate semantics manager.

  Returns:
      Configured `isolate_semantics_manager` fixture value (typing.Generator[None, None, None]).
  """
  from ml_switcheroo.semantics import manager as manager_module

  original_cache = manager_module._MANAGER_CACHE
  manager_module._MANAGER_CACHE = None
  yield
  manager_module._MANAGER_CACHE = original_cache


@pytest.fixture(autouse=True)
def isolate_hook_registry() -> typing.Generator[None, None, None]:
  """Helper to isolate hook registry.

  Returns:
      Configured `isolate_hook_registry` fixture value (typing.Generator[None, None, None]).
  """
  import ml_switcheroo.core.hooks as hooks_module
  import ml_switcheroo.core.hooks_registry as hr_module
  from ml_switcheroo.core.hooks import _HOOK_METADATA, _HOOKS

  original_hooks = _HOOKS.copy()
  original_metadata = _HOOK_METADATA.copy()
  original_loaded = getattr(hr_module, "_PLUGINS_LOADED", False)
  yield
  _HOOKS.clear()
  _HOOKS.update(original_hooks)
  _HOOK_METADATA.clear()
  _HOOK_METADATA.update(original_metadata)
  hr_module._PLUGINS_LOADED = original_loaded
  hooks_module._PLUGINS_LOADED = original_loaded


@pytest.fixture(autouse=True)
def isolate_framework_registry() -> typing.Generator[None, None, None]:
  """Helper to isolate framework registry.

  Returns:
      Configured `isolate_framework_registry` fixture value (typing.Generator[None, None, None]).
  """
  original_registry = _ADAPTER_REGISTRY.copy()
  yield
  _ADAPTER_REGISTRY.clear()
  _ADAPTER_REGISTRY.update(original_registry)


def pytest_addoption(parser: pytest.Parser) -> None:
  """Execute pytest addoption helper logic.

  Args:
      parser: Parser instance under test.
  """
  parser.addoption("--update-snapshots", action="store_true", default=False, help="Update snapshots for visual tests")
