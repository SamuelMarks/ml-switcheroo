"""Auto-generated doc."""

from ml_switcheroo.core.compiler.frontends.sass.parser import SassParser, LabelRef


def test_sass_parser_missing():
  """Auto-generated doc."""
  parser = SassParser(".text\n.global main")
  r = LabelRef("test")
  assert str(r) == "test"

  parser.parse()

  parser = SassParser("MOV R0, R1\n.text")
  parser.parse()
