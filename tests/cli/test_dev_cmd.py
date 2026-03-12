import pytest
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock

from ml_switcheroo.cli.handlers.dev import handle_matrix, handle_docs, handle_gen_tests


def test_handle_matrix():
  with (
    patch("ml_switcheroo.cli.handlers.dev.SemanticsManager"),
    patch("ml_switcheroo.cli.handlers.dev.CompatibilityMatrix") as MockMatrix,
  ):
    res = handle_matrix()
    assert res == 0
    MockMatrix.return_value.render.assert_called_once()


def test_handle_docs():
  with (
    patch("ml_switcheroo.cli.handlers.dev.SemanticsManager"),
    patch("ml_switcheroo.cli.handlers.dev.MigrationGuideGenerator") as MockGen,
    patch("builtins.open", mock_open()) as m_open,
  ):
    MockGen.return_value.generate.return_value = "MARKDOWN"
    res = handle_docs("jax", "torch", Path("out.md"))
    assert res == 0
    m_open.assert_called_once_with(Path("out.md"), "w", encoding="utf-8")
    m_open().write.assert_called_once_with("MARKDOWN")


def test_handle_gen_tests():
  with (
    patch("ml_switcheroo.cli.handlers.dev.SemanticsManager") as MockSemantics,
    patch("ml_switcheroo.cli.handlers.dev.TestCaseGenerator") as MockGen,
    patch("pathlib.Path.mkdir"),
  ):
    MockSemantics.return_value.get_known_apis.return_value = {}
    res = handle_gen_tests(Path("tests/out"))
    assert res == 0
    MockGen.return_value.generate.assert_called_once()
