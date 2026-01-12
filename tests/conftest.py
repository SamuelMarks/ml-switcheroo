import sys
import pytest
import warnings
from pathlib import Path
from typing import Callable, Optional

# --- FIX: Global Warning Suppression for Collection Phase ---
# Suppress the Keras/NumPy 'np.object' FutureWarning that crashes collection
warnings.filterwarnings("ignore", message=".*np\\.object.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="keras.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="tensorflow.*")
# ------------------------------------------------------------

# Add src to path so we can import 'ml_switcheroo' without installing it
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Force load of default adapters so they provide the "clean state" baseline
try:
  import ml_switcheroo.frameworks
  from ml_switcheroo.frameworks.base import _ADAPTER_REGISTRY
except ImportError:
  _ADAPTER_REGISTRY = {}

# Import Rewriter components for TestRewriter shim
from ml_switcheroo.core.rewriter import (
  RewriterContext,
  RewriterPipeline,
  StructuralPass,
  ApiPass,
  AuxiliaryPass,
)


class TestRewriter:
  """
  Test-scoped shim replacing the legacy PivotRewriter.

  Wraps the RewriterPipeline to allow tests to execute transformations
  deterministically without the full Engine overhead. It mimics the
  interface used by legacy tests while using the modern pipeline architecture.
  """

  # Prevent pytest from trying to collect this as a test class containing tests
  __test__ = False

  def __init__(self, semantics, config, symbol_table=None):
    self.context = RewriterContext(semantics, config, symbol_table)
    self.pipeline = RewriterPipeline([StructuralPass(), ApiPass(), AuxiliaryPass()])

  @property
  def ctx(self):
    """Access hook context for tests that inspect internal state."""
    return self.context.hook_context

  @property
  def semantics(self):
    """Access semantics manager from context."""
    return self.context.semantics

  def convert(self, module):
    """
    Executes the pipeline on the given CST module.
    Replaces the old pattern `tree.visit(rewriter)`.
    """
    return self.pipeline.run(module, self.context)


class SnapshotAssert:
  """
  Simple snapshot comparison logic to verify CLI output stability.
  """

  def __init__(self, request: pytest.FixtureRequest):
    self.request = request
    self.test_name = request.node.name
    self.module_path = Path(request.node.fspath).parent
    self.snapshot_dir = self.module_path / "__snapshots__"
    self.update_mode = request.config.getoption("--update-snapshots", default=False)

  def assert_match(self, content: str, extension: str = "txt", normalizer: Optional[Callable[[str], str]] = None):
    """
    Compares content against stored file.

    Args:
        content: The actual output string.
        extension: File extension (default 'txt', 'json', etc).
        normalizer: Optional function to clean both content and expected string before comparison.
    """
    if not self.snapshot_dir.exists():
      self.snapshot_dir.mkdir(parents=True)

    snapshot_file = self.snapshot_dir / f"{self.test_name}.{extension}"

    # Normalize line endings
    content = content.replace("\r\n", "\n")

    if self.update_mode or not snapshot_file.exists():
      # Write raw content to disk
      normalized_to_write = normalizer(content) if normalizer else content
      snapshot_file.write_text(normalized_to_write, encoding="utf-8")
      if self.update_mode:
        return

    expected = snapshot_file.read_text(encoding="utf-8").replace("\r\n", "\n")

    lhs = content
    rhs = expected

    if normalizer:
      lhs = normalizer(lhs)
      rhs = normalizer(rhs)

    assert lhs == rhs, f"Snapshot mismatch for {snapshot_file.name}. Run with --update-snapshots to accept changes."


@pytest.fixture
def snapshot(request):
  """Fixture to assert text matches a stored snapshot."""
  return SnapshotAssert(request)


@pytest.fixture(autouse=True)
def isolate_framework_registry():
  """
  Ensures that modifications to the framework adapter registry
  (adding custom frameworks for tests) do not leak between tests.
  """
  original_registry = _ADAPTER_REGISTRY.copy()
  yield
  _ADAPTER_REGISTRY.clear()
  _ADAPTER_REGISTRY.update(original_registry)


def pytest_addoption(parser):
  """Add CLI flag to update snapshots."""
  parser.addoption("--update-snapshots", action="store_true", default=False, help="Update snapshots for visual tests")
