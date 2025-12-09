"""
Pytest Configuration and Fixtures.

Includes:
- Syspath patching for local imports.
- Snapshot testing fixture for visual verification.
- Global registry isolation to prevent tests with custom adapters from leaking.
"""

import sys
import pytest
from pathlib import Path
from typing import Callable, Optional

# Add src to path so we can import 'ml_switcheroo' without installing it
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Force load of default adapters so they provide the "clean state" baseline
# This ensures that standard frameworks (Torch, JAX, etc.) are present in the
# snapshot we take for isolation, so they remain available after restoration.
try:
  import ml_switcheroo.frameworks
  from ml_switcheroo.frameworks.base import _ADAPTER_REGISTRY
except ImportError:
  _ADAPTER_REGISTRY = {}


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
      # We write the raw content (without external normalization) to disk
      # so the file remains human-readable/representative of raw output if preferred,
      # or we could write normalized. Usually better to write raw and normalize on read
      # for comparison robustness.
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

    # Simple assertion. Diff tools in IDE handles failures well.
    assert lhs == rhs, (
      f"Snapshot mismatch for {snapshot_file.name}. Run check script or pytest with --update-snapshots to accept changes."
    )


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
  # Capture the state (which ideally contains torch, jax, etc. loaded via imports above)
  original_registry = _ADAPTER_REGISTRY.copy()
  yield
  # Restore state after test completion
  _ADAPTER_REGISTRY.clear()
  _ADAPTER_REGISTRY.update(original_registry)


def pytest_addoption(parser):
  """Add CLI flag to update snapshots."""
  parser.addoption("--update-snapshots", action="store_true", default=False, help="Update snapshots for visual tests")
