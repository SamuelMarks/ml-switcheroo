from ml_switcheroo.compiler.frontends.sass.parser import SassParser


def test_sass_parser_missing():
  parser = SassParser(".text\n.global main")
  try:
    from ml_switcheroo.compiler.frontends.sass.parser import LabelRef as PLabelRef

    r = PLabelRef("test")
    assert str(r) == "test"
  except ImportError:
    pass

  ast = parser.parse()

  parser = SassParser("MOV R0, R1\n.text")
  ast = parser.parse()
