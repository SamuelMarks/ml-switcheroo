"""
Tests for Codebase Auditing Tools.

This module verifies the functionality of the `audit` command, which scans
source code to identify operations that are supported (mapped) or missing
(unmapped) in the Knowledge Base.

It covers:
1.  **Static Analysis**: `CoverageScanner` resolving imports, aliases, and attribute calls.
2.  **Root Resolution**: Expanding framework keys (e.g., `flax_nnx`) to module roots (e.g., `flax`).
3.  **CLI Interaction**: Validating that `handle_audit` produces the correct table output and exit codes.
"""

import pytest
import libcst as cst
from unittest.mock import MagicMock, patch
from pathlib import Path
from typing import Optional, Dict

from ml_switcheroo.analysis.audit import CoverageScanner
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.cli.handlers.audit import handle_audit, resolve_roots


class MockAuditSemantics(SemanticsManager):
  """
  Mock Semantics Manager for Audit tests.

  Bypasses file loading and provides a deterministic lookup dictionary.
  """

  def __init__(self, known_apis: Dict[str, str]):
    """
    Initialize with a specific set of known APIs.

    Args:
        known_apis: Dict mapping FQN -> Framework Key (e.g., {'torch.abs': 'torch'}).
    """
    # Bypass super init to avoid file I/O
    self.known = known_apis

  def get_definition(self, api_name: str) -> Optional[tuple]:
    """
    Simulates looking up an API definition.

    Args:
        api_name: The fully qualified name (e.g. 'torch.abs').

    Returns:
        Tuple (id, details) if found, else None.
    """
    if api_name in self.known:
      fw_key = self.known[api_name]
      # Construct minimal variant structure
      return "op_id", {"variants": {fw_key: {"api": api_name}}}
    return None


@pytest.fixture
def semantics() -> MockAuditSemantics:
  """
  Returns a mocked SemanticsManager with known APIs for 'torch' and 'jax'.
  """
  return MockAuditSemantics({"torch.abs": "torch", "jax.numpy.sum": "jax"})


def scan_code(code: str, semantics: SemanticsManager, roots: list[str]) -> Dict:
  """
  Helper to parse code and run the CoverageScanner.

  Args:
      code: Python source string.
      semantics: The SemanticsManager instance.
      roots: List of allowed framework roots.

  Returns:
      The `scanner.results` dictionary.
  """
  module = cst.parse_module(code)
  scanner = CoverageScanner(semantics, set(roots))
  module.visit(scanner)
  return scanner.results


# --- Analysis Logic Tests ---


def test_scanner_direct_hit(semantics):
  """
  Scenario: Source uses a supported API without aliasing.
  Expectation: API is found, marked as supported, framework detected correctly.
  """
  code = "y = torch.abs(x)"
  results = scan_code(code, semantics, roots=["torch"])

  assert "torch.abs" in results
  is_supported, fw = results["torch.abs"]
  assert is_supported is True
  assert fw == "torch"


def test_scanner_alias_resolution_jnp(semantics):
  """
  Scenario: Source uses standard alias `import jax.numpy as jnp`.
  Expectation: `jnp.sum` resolves to `jax.numpy.sum` and is marked supported.
  """
  code = """
import jax.numpy as jnp
y = jnp.sum(x)
"""
  results = scan_code(code, semantics, roots=["jax"])

  assert "jax.numpy.sum" in results
  is_supported, fw = results["jax.numpy.sum"]
  assert is_supported is True
  assert fw == "jax"


def test_scanner_missing_operation(semantics):
  """
  Scenario: Source uses an API not present in the Semantics Manager.
  Expectation: API is found but marked unsupported (False).
  """
  code = "y = torch.unknown_func(x)"
  results = scan_code(code, semantics, roots=["torch"])

  assert "torch.unknown_func" in results
  is_supported, fw = results["torch.unknown_func"]
  assert is_supported is False
  # Framework fallback guessing from root
  assert fw == "torch"


def test_scanner_ignore_irrelevant_roots(semantics):
  """
  Scenario: Source uses `os.path.join`.
  Expectation: Ignored because 'os' is not in allowed roots.
  """
  code = """
import os
p = os.path.join('a', 'b')
"""
  results = scan_code(code, semantics, roots=["torch", "jax"])
  assert len(results) == 0


def test_scanner_from_import_resolution(semantics):
  """
  Scenario: `from jax.numpy import sum as s`.
  Expectation: `s(x)` resolves to `jax.numpy.sum`.
  """
  code = """
from jax.numpy import sum as s
y = s(x)
"""
  results = scan_code(code, semantics, roots=["jax"])

  assert "jax.numpy.sum" in results
  assert results["jax.numpy.sum"][0] is True


# --- CLI Handler Tests ---


def test_resolve_roots_expansion():
  """
  Scenario: User requests audit for 'flax_nnx'.
  Expectation: Returns set containing {'flax_nnx', 'flax', 'jax'} based on mock adapter metadata.
  """
  mock_adapter = MagicMock()
  mock_adapter.import_alias = ("flax.nnx", "nnx")
  mock_adapter.search_modules = ["jax.numpy"]

  with patch("ml_switcheroo.cli.handlers.audit.get_adapter", return_value=mock_adapter):
    roots = resolve_roots(["flax_nnx"])

  assert "flax" in roots  # From import alias
  assert "jax" in roots  # From search modules
  assert "flax_nnx" in roots  # Original key preserved


def test_handle_audit_missing_file(tmp_path):
  """
  Scenario: Path argument does not exist.
  Expectation: Log error, return exit code 1.
  """
  bad_path = tmp_path / "ghost.py"
  ret = handle_audit(bad_path, ["torch"])
  assert ret == 1


def test_handle_audit_clean_report(tmp_path, capsys):
  """
  Scenario: Code contains only supported operations.
  Expectation:
      - Exit code 0.
      - Summary printed.
      - Missing Operations table NOT printed.
  """
  f = tmp_path / "clean.py"
  f.write_text("import torch\ny = torch.abs(x)")

  # Mock manager to recognize torch.abs
  mock_mgr_cls = MagicMock(return_value=MockAuditSemantics({"torch.abs": "torch"}))

  with patch("ml_switcheroo.cli.handlers.audit.SemanticsManager", mock_mgr_cls):
    ret = handle_audit(f, ["torch"])

  assert ret == 0
  captured = capsys.readouterr()
  assert "Audit Summary" in captured.out
  assert "Supported:         1" in captured.out

  # Assert that the FAILURE TABLE is NOT present
  assert "❌ Missing Operations" not in captured.out

  # "Missing: 0" line IS present in the summary, so we do NOT assert its absence
  assert "Missing:" in captured.out


def test_handle_audit_missing_report(tmp_path, capsys):
  """
  Scenario: Code contains unsupported operations.
  Expectation:
      - Exit code 1.
      - Table printed with 'API Name' and 'Framework' columns.
  """
  f = tmp_path / "dirty.py"
  f.write_text("import torch\ny = torch.secret_op(x)")

  # Mock manager knows nothing
  mock_mgr_cls = MagicMock(return_value=MockAuditSemantics({}))

  with patch("ml_switcheroo.cli.handlers.audit.SemanticsManager", mock_mgr_cls):
    ret = handle_audit(f, ["torch"])

  assert ret == 1
  captured = capsys.readouterr()

  # Check Table Headers specific to the new design
  assert "Framework" in captured.out
  assert "API Name" in captured.out

  # Check Row Content
  assert "torch" in captured.out
  assert "torch.secret_op" in captured.out
