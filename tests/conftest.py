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
from ml_switcheroo.core.hooks import HookContext, get_default_hook_registry
from ml_switcheroo.core.rewriter import ApiPass, AuxiliaryPass, RewriterContext, RewriterPipeline, StructuralPass
from ml_switcheroo.core.rewriter.context import SemanticsProtocol
from ml_switcheroo.frameworks.base import get_default_adapter_registry
from ml_switcheroo.semantics.manager import SemanticsManager, clear_semantics_cache

warnings.filterwarnings("ignore", message=".*np\\.object.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="keras.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="tensorflow.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow\\.lite.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="jax\\._src\\.deprecations.*")
src_path: Path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))
importlib.import_module("ml_switcheroo.frameworks")


class TestRewriter:
  """Lightweight pipeline wrapper running Structural, API, and Auxiliary passes for tests."""

  __test__ = False

  def __init__(
    self,
    semantics: SemanticsManager | SemanticsProtocol,
    config: RuntimeConfig | None = None,
    symbol_table: SymbolTable | None = None,
  ) -> None:
    """Initialize the rewriter wrapper with semantics, runtime config, and symbol table.

    Args:
        semantics: SemanticsManager instance providing operation and trait definitions.
        config: RuntimeConfig specifying source and target frameworks.
        symbol_table: SymbolTable tracking variable types and aliases.
    """
    self.context = RewriterContext(semantics, config, symbol_table)
    self.pipeline = RewriterPipeline([StructuralPass(), ApiPass(), AuxiliaryPass()])

  @property
  def ctx(self) -> HookContext:
    """Return the active `HookContext` bound to the underlying rewriter context.

    Returns:
        Active `HookContext` instance.
    """
    return self.context.hook_context

  @property
  def semantics(self) -> SemanticsManager:
    """Return the `SemanticsManager` backing this rewriter instance.

    Returns:
        Configured `SemanticsManager` instance.
    """
    return self.context.semantics

  def convert(self, module: cst.Module) -> cst.Module:
    """Run the structural, API, and auxiliary rewrite passes over `module`.

    Args:
        module: Parsed LibCST module to transform.

    Returns:
        Transformed LibCST module after all passes execute.
    """
    return self.pipeline.run(module, self.context)


class SnapshotAssert:
  """Golden file snapshot comparator supporting interactive `--update-snapshots` refreshes."""

  def __init__(self, request: pytest.FixtureRequest) -> None:
    """Initialize snapshot paths and update flags from the pytest request context.

    Args:
        request: Pytest fixture request context.
    """
    self.request = request
    self.test_name: str = request.node.name
    self.module_path: Path = Path(request.node.fspath).parent
    self.snapshot_dir: Path = self.module_path / "__snapshots__"
    self.update_mode: bool = request.config.getoption("--update-snapshots", default=False)

  @staticmethod
  def _normalize_posix_text(text: str) -> str:
    """Normalize line endings, strip trailing horizontal whitespace, and ensure a final newline.

    Args:
        text: Raw snapshot text to normalize.

    Returns:
        POSIX-normalized text matching `trailing-whitespace` and `end-of-file-fixer` hooks.
    """
    normalized = "\n".join(line.rstrip(" \t") for line in text.replace("\r\n", "\n").split("\n"))
    if not normalized.strip("\n"):
      return ""
    return normalized.rstrip("\n") + "\n"

  def assert_match(
    self, content: str, extension: str = "txt", normalizer: typing.Callable[[str], str] | None = None
  ) -> None:
    """Compare `content` against the stored snapshot file or write it in update mode.

    Args:
        content: Rendered text output produced by the test under verification.
        extension: File extension for the snapshot artifact without a leading dot.
        normalizer: Optional string normalization function applied prior to comparison.
    """
    if not self.snapshot_dir.exists():
      self.snapshot_dir.mkdir(parents=True)
    snapshot_file: Path = self.snapshot_dir / f"{self.test_name}.{extension}"
    lhs: str = self._normalize_posix_text(normalizer(content) if normalizer else content)
    if self.update_mode or not snapshot_file.exists():
      snapshot_file.write_text(lhs, encoding="utf-8")
      if self.update_mode:
        return
    expected: str = snapshot_file.read_text(encoding="utf-8")
    rhs: str = self._normalize_posix_text(normalizer(expected) if normalizer else expected)
    assert lhs == rhs, f"Snapshot mismatch for {snapshot_file.name}. Run with --update-snapshots to accept changes."


@pytest.fixture
def snapshot(request: pytest.FixtureRequest) -> SnapshotAssert:
  """Create a `SnapshotAssert` helper bound to the current pytest node.

  Args:
      request: Pytest fixture request context.

  Returns:
      Configured `SnapshotAssert` helper instance.
  """
  return SnapshotAssert(request)


@pytest.fixture(autouse=True)
def isolate_semantics_manager() -> typing.Generator[None, None, None]:
  """Clear the default ``SemanticsManager`` ``lru_cache`` before and after each test.

  Yields:
      None while executing the isolated test case.
  """
  clear_semantics_cache()
  yield
  clear_semantics_cache()


@pytest.fixture(autouse=True)
def isolate_hook_registry() -> typing.Generator[None, None, None]:
  """Snapshot and restore the default ``HookRegistry`` around each test.

  Yields:
      None while executing the isolated test case.
  """
  registry = get_default_hook_registry()
  saved = registry.snapshot()
  yield
  registry.restore(saved)


@pytest.fixture(autouse=True)
def isolate_framework_registry() -> typing.Generator[None, None, None]:
  """Snapshot and restore the default ``AdapterRegistry`` around each test.

  Yields:
      None while executing the isolated test case.
  """
  registry = get_default_adapter_registry()
  saved = registry.snapshot()
  yield
  registry.restore(saved)


def pytest_addoption(parser: pytest.Parser) -> None:
  """Register the `--update-snapshots` CLI flag with pytest.

  Args:
      parser: Pytest command-line option parser.
  """
  parser.addoption("--update-snapshots", action="store_true", default=False, help="Update snapshots for visual tests")
