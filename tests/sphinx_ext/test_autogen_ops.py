"""
Tests for the Autogen Ops Sphinx Hook.

Verifies:
1.  Recursion of SemanticsManager logic inside the hook.
2.  File creation in the `docs/ops` directory.
3.  Structure of the `index.rst` table of contents.
4.  Robustness against missing data.
"""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from ml_switcheroo.sphinx_ext.autogen_ops import generate_op_docs


@pytest.fixture
def mock_app(tmp_path):
  """
  Mocks the Sphinx Application object, providing a `srcdir` that
  points to a temporary directory.
  """
  app = MagicMock()
  # Setup srcdir to simulate docs/ folder
  docs_dir = tmp_path / "docs"
  docs_dir.mkdir()
  app.srcdir = str(docs_dir)
  return app


def test_autogen_creates_files(mock_app):
  """
  Scenario: Semantics contain 1 Op ('TestOp').
  Expectation: 'ops/TestOp.rst' and 'ops/index.rst' are created.
  """
  # Mock Semantics
  mock_apis = {"TestOp": {"description": "A test op.", "std_args": ["x"], "variants": {"torch": {"api": "torch.foo"}}}}

  with patch("ml_switcheroo.sphinx_ext.autogen_ops.SemanticsManager") as MockMgr:
    mgr = MockMgr.return_value
    mgr.get_known_apis.return_value = mock_apis

    # Run Hook
    generate_op_docs(mock_app)

  # Verify Output Directory
  ops_dir = Path(mock_app.srcdir) / "ops"
  assert ops_dir.exists()

  # Verify Op File
  op_file = ops_dir / "TestOp.rst"
  assert op_file.exists()
  content = op_file.read_text("utf-8")
  assert "TestOp" in content
  assert "A test op." in content
  assert "Standard Signature" in content or "Abstract Signature" in content

  # Verify Index File
  index_file = ops_dir / "index.rst"
  assert index_file.exists()
  index_content = index_file.read_text("utf-8")
  assert "Operation Reference" in index_content
  assert "TestOp" in index_content


def test_autogen_handles_empty_semantics(mock_app, caplog):
  """
  Scenario: Semantics are empty.
  Expectation: Directory created, index created (empty list), warning logged.
  """
  with patch("ml_switcheroo.sphinx_ext.autogen_ops.SemanticsManager") as MockMgr:
    mgr = MockMgr.return_value
    mgr.get_known_apis.return_value = {}

    generate_op_docs(mock_app)

  ops_dir = Path(mock_app.srcdir) / "ops"
  assert ops_dir.exists()

  # Check warning log
  assert "No operations found" in caplog.text

  # Index should still exist but have no entries
  index_file = ops_dir / "index.rst"
  assert index_file.exists()


def test_autogen_sanitizes_filenames(mock_app):
  """
  Scenario: Operation name contains weird characters 'My Op/1'.
  Expectation: Filename is sanitized to 'MyOp1.rst'.
  """
  mock_apis = {"My Op/1": {"description": "Complex name", "std_args": [], "variants": {}}}

  with patch("ml_switcheroo.sphinx_ext.autogen_ops.SemanticsManager") as MockMgr:
    mgr = MockMgr.return_value
    mgr.get_known_apis.return_value = mock_apis
    generate_op_docs(mock_app)

  ops_dir = Path(mock_app.srcdir) / "ops"

  expected_file = ops_dir / "MyOp1.rst"
  assert expected_file.exists()
