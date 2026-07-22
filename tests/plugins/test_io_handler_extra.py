"""Test suite for the Io Handler Extra module."""

import libcst as cst
from ml_switcheroo.plugins.io_handler import _get_arg


def test_get_arg_wrong_keyword():
  """Gets argument wrong keyword."""
  arg = cst.Arg(value=cst.Name("val"), keyword=cst.Name("wrong_name"))
  assert _get_arg([arg], 0, "obj") is None
