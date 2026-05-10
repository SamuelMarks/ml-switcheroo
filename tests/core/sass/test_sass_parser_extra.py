from ml_switcheroo.compiler.frontends.sass.parser import SassParser, LabelRef


def test_sass_parser_missing():
  parser = SassParser(".text\n.global main")
  r = LabelRef("test")
  assert str(r) == "test"

  parser.parse()

  parser = SassParser("MOV R0, R1\n.text")
  parser.parse()
