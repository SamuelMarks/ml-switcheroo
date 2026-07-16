"""Auto-generated doc."""

from ml_switcheroo.core.compiler.frontends.sass.parser import SassParser


def test_sass_parser_missing():
  """Auto-generated doc."""
  parser = SassParser(".text\n.global main")
  try:
    from ml_switcheroo.core.compiler.frontends.sass.nodes import LabelRef as PLabelRef

    r = PLabelRef("test")
    assert str(r) == "test"
  except ImportError:
    pass

  parser.parse()

  parser = SassParser("MOV R0, R1\n.text")
  parser.parse()
