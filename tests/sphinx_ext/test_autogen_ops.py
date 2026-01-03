"""
Tests for the Autogen Ops Sphinx Hook.

Verifies:
1.  Recursion of SemanticsManager logic inside the hook.
2.  File creation in the `docs/ops` directory.
3.  Structure of the `index.rst` table of contents.
4.  Filtering of Orphaned operations (implemented by < 2 frameworks).
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


def test_autogen_creates_files_for_valid_ops(mock_app):
  """
  Scenario: Semantics contain 1 Op implemented by 2 frameworks.
  Expectation: 'ops/ValidOp.rst' and 'ops/index.rst' are created.
  """
  # Mock Semantics
  mock_apis = {
    "ValidOp": {
      "description": "A valid op.",
      "std_args": ["x"],
      "variants": {"torch": {"api": "torch.foo"}, "jax": {"api": "jnp.foo"}},
    }
  }

  with patch("ml_switcheroo.sphinx_ext.autogen_ops.SemanticsManager") as MockMgr:
    mgr = MockMgr.return_value
    mgr.get_known_apis.return_value = mock_apis

    # Run Hook
    generate_op_docs(mock_app)

  # Verify Output Directory
  ops_dir = Path(mock_app.srcdir) / "ops"
  assert ops_dir.exists()

  # Verify Op File Created
  op_file = ops_dir / "ValidOp.rst"
  assert op_file.exists()
  content = op_file.read_text("utf-8")
  assert "ValidOp" in content

  # Verify Index File Created and Linked
  index_file = ops_dir / "index.rst"
  assert index_file.exists()
  index_content = index_file.read_text("utf-8")
  assert "ValidOp" in index_content


def test_autogen_skips_orphan_ops(mock_app, caplog):
  """
  Scenario: Semantics contain 1 Op implemented by ONLY 1 framework (Orphan).
  Expectation: Directory created, but NO operation file created.
  """
  mock_apis = {
    "OrphanOp": {
      "description": "A lonely op.",
      "std_args": ["x"],
      "variants": {
        "torch": {"api": "torch.foo"},
        "jax": None,  # Explicitly disabled
      },
    }
  }

  with patch("ml_switcheroo.sphinx_ext.autogen_ops.SemanticsManager") as MockMgr:
    mgr = MockMgr.return_value
    mgr.get_known_apis.return_value = mock_apis
    generate_op_docs(mock_app)

  ops_dir = Path(mock_app.srcdir) / "ops"

  # File should NOT exist
  op_file = ops_dir / "OrphanOp.rst"
  assert not op_file.exists()

  # Index should NOT link it
  index_file = ops_dir / "index.rst"
  index_content = index_file.read_text("utf-8")
  assert "OrphanOp" not in index_content


def test_autogen_handles_empty_semantics(mock_app, caplog):
  """
  Scenario: Semantics are completely empty.
  Expectation: Warning logged, no crashes.
  """
  with patch("ml_switcheroo.sphinx_ext.autogen_ops.SemanticsManager") as MockMgr:
    mgr = MockMgr.return_value
    mgr.get_known_apis.return_value = {}

    generate_op_docs(mock_app)

  ops_dir = Path(mock_app.srcdir) / "ops"
  assert ops_dir.exists()

  # Check warning log
  assert "No operations found" in caplog.text
