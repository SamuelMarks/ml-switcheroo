"""Test suite for the Sass Parser Coverage module."""

import pytest
from ml_switcheroo.core.compiler.frontends.sass.parser import SassParser
from ml_switcheroo.core.compiler.frontends.sass.cst import SassMemory, SassRegister


def test_sass_parser_missing():
  """Verifies the behavior of SASS parser missing."""
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


def test_sass_parser_error():
  """Test for test_sass_parser_error."""
  with pytest.raises(ValueError, match="Unexpected"):
    SassParser("~").parse()

  with pytest.raises(ValueError):
    SassParser("MOV ¿").parse()


def test_sass_parser_empty():
  """Test for test_sass_parser_empty."""
  mod = SassParser("   ").parse()
  assert len(mod.statements) == 0


def test_sass_parser_directive_no_params():
  """Test for test_sass_parser_directive_no_params."""
  mod = SassParser(".text").parse()
  assert mod.statements[0].name == "text"


def test_sass_parser_directive_string():
  """Test for test_sass_parser_directive_string."""
  mod = SassParser('.headerflags @"NV_PROFILE"').parse()
  assert mod.statements[0].params[0] == '@"NV_PROFILE"'

  mod = SassParser('.headerflags "NV_PROFILE"').parse()
  assert mod.statements[0].params[0] == '"NV_PROFILE"'


def test_sass_parser_comments():
  """Test for test_sass_parser_comments."""
  mod = SassParser("// foo").parse()
  assert mod.statements[0].text == "foo"


def test_sass_parser_empty_stmt():
  """Test for test_sass_parser_empty_stmt."""
  mod = SassParser(";").parse()
  assert len(mod.statements) == 0


def test_sass_parser_mem_reg():
  """Test for test_sass_parser_mem_reg."""
  mod = SassParser("LDG R0, [R1]").parse()
  assert getattr(mod.statements[0].operands[1], "offset", "fake") is None


def test_sass_parser_predicate_operand():
  """Test for test_sass_parser_predicate_operand."""
  mod = SassParser("ISETP.NE.AND P0, PT, R1, 0x0, PT").parse()
  assert mod.statements[0].operands[-1].name == "PT"


def test_sass_parser_at_string():
  """Test for test_sass_parser_at_string."""
  mod = SassParser('.headerflags @"foo"').parse()
  assert mod.statements[0].params[0] == '@"foo"'


def test_sass_parser_immediate():
  """Test for test_sass_parser_immediate."""
  mod = SassParser("MOV R0, 5").parse()
  assert mod.statements[0].operands[1].value == 5
  assert mod.statements[0].operands[1].is_hex is False

  mod = SassParser("MOV R0, 5.0").parse()
  assert mod.statements[0].operands[1].value == 5.0
  assert mod.statements[0].operands[1].is_hex is False


def test_sass_parser_register():
  """Test for test_sass_parser_register."""
  mod = SassParser("MOV R0, -R1").parse()
  assert mod.statements[0].operands[1].negated is True

  mod = SassParser("MOV R0, |-R1|").parse()
  assert mod.statements[0].operands[1].absolute is True
  assert mod.statements[0].operands[1].negated is True


def test_sass_parser_mem_bank():
  """Test for test_sass_parser_mem_bank."""
  mod = SassParser("FADD R0, R1, c[0x0][0x4]").parse()
  assert mod.statements[0].operands[2].offset == 4


def test_sass_parser_mem_offset():
  """Test for test_sass_parser_mem_offset."""
  mod = SassParser("LDG.E R0, [R1 + 0x8]").parse()
  assert mod.statements[0].operands[1].offset == 8

  mod = SassParser("LDG.E R0, [R1 + 8]").parse()
  assert mod.statements[0].operands[1].offset == 8


def test_sass_parser_predicate_guard():
  """Test for test_sass_parser_predicate_guard."""
  mod = SassParser("@!P0 MOV R0, R1").parse()
  assert mod.statements[0].predicate.negated is True
  assert mod.statements[0].predicate.name == "P0"


def test_missing_coverage_sass():
  """Test for test_missing_coverage_sass."""
  # label
  mod = SassParser("main:").parse()
  assert mod.statements[0].name == "main"

  # mem reg only
  mod = SassParser("LDG R0, [R1]").parse()
  assert mod.statements[0].operands[1].offset is None

  # predicate operand with negation
  mod = SassParser("ISETP.NE.AND P0, PT, R1, 0x0, !PT").parse()
  assert mod.statements[0].operands[-1].negated is True

  # operands single
  mod = SassParser("RET").parse()
  assert len(mod.statements[0].operands) == 0

  # unreached token mismatch
  with pytest.raises(ValueError, match="Unexpected"):
    SassParser("?").parse()


def test_missing_coverage_sass_2():
  """Test for test_missing_coverage_sass_2."""
  # directive list param
  mod = SassParser('.headerflags @"NV_PROFILE", "OTHER"').parse()
  assert len(mod.statements[0].params) == 2

  mod = SassParser('.headerflags @"NV_PROFILE", @"OTHER"').parse()
  assert mod.statements[0].params[1] == '@"OTHER"'


def test_missing_coverage_sass_3():
  """Test for test_missing_coverage_sass_3."""
  # directive param fallback list and Token fallback
  mod = SassParser(".req 5").parse()  # number identifier?
  assert mod.statements[0].params[0] == "5"

  mod = SassParser('.req foo, 5, @"test"').parse()
  assert len(mod.statements[0].params) == 3


def test_missing_coverage_sass_4():
  """Test for test_missing_coverage_sass_4."""
  # label trivia
  mod = SassParser("main:\n  MOV R0, R1").parse()
  assert mod.statements[0].name == "main"

  # predicate missing coverage (no exclamation)
  mod = SassParser("@P0 MOV R0, R1").parse()
  assert mod.statements[0].predicate.negated is False

  # mem reg fallback
  mod = SassParser("LDG R0, [R1]").parse()
  assert mod.statements[0].operands[1].offset is None


def test_missing_coverage_sass_5():
  """Test for test_missing_coverage_sass_5."""
  # label operand
  mod = SassParser("BRA main").parse()
  assert mod.statements[0].operands[0].name == "main"


def test_missing_coverage_sass_6():
  """Test for test_missing_coverage_sass_6."""
  # at_string parsing direct string value fallback
  mod = SassParser('.headerflags "NV_PROFILE"').parse()
  assert mod.statements[0].params[0] == '"NV_PROFILE"'


def test_missing_coverage_sass_7():
  """Test for test_missing_coverage_sass_7."""
  mod = SassParser(".headerflags 5").parse()
  assert mod.statements[0].params[0] == "5"

  mod = SassParser("MOV R0").parse()
  assert len(mod.statements[0].operands) == 1

  mod = SassParser("main:").parse()
  assert mod.statements[0].name == "main"


def test_sass_parser_pred_at_bang_id():
  """Test for test_sass_parser_pred_at_bang_id."""
  mod = SassParser("FADD R0, @!P0").parse()
  assert mod.statements[0].operands[1].negated is True
  assert mod.statements[0].operands[1].name == "P0"


def test_sass_parser_pred_at_id():
  """Test for test_sass_parser_pred_at_id."""
  mod = SassParser("FADD R0, @P0").parse()
  assert mod.statements[0].operands[1].negated is False


def test_sass_parser_pred_bang_reg():
  """Test for test_sass_parser_pred_bang_reg."""
  mod = SassParser("FADD R0, !PT").parse()
  assert mod.statements[0].operands[1].negated is True


def test_sass_parser_pred_reg():
  """Test for test_sass_parser_pred_reg."""
  mod = SassParser("FADD R0, PT").parse()
  assert isinstance(mod.statements[0].operands[1], SassRegister)


def test_sass_parser_mem_bank_2():
  """Test for test_sass_parser_mem_bank_2."""
  mod = SassParser("FADD R0, c[0x0][0x4]").parse()
  assert isinstance(mod.statements[0].operands[1], SassMemory)
  assert mod.statements[0].operands[1].offset == 4


def test_sass_parser_mem_reg_neg_offset():
  """Test for test_sass_parser_mem_reg_neg_offset."""
  mod = SassParser("FADD R0, [R1 - 0x4]").parse()
  assert mod.statements[0].operands[1].offset == -4


def test_sass_parser_missing_coverage_8():
  """Test for test_sass_parser_missing_coverage_8."""
  mod = SassParser("FADD R0, @!R1").parse()
  assert mod.statements[0].operands[1].negated is True

  mod = SassParser("FADD R0, @R1").parse()
  assert mod.statements[0].operands[1].negated is False

  mod = SassParser("FADD R0, !P1").parse()
  assert mod.statements[0].operands[1].negated is True


def test_sass_parser_missing_coverage_9():
  """Test for test_sass_parser_missing_coverage_9."""
  # hit line 104 (node missing leading_trivia but has children)
  from ml_switcheroo.core.compiler.frontends.sass.parser import _get_trivia

  class DummyChild:
    """A dummy child node."""

    def __init__(self):
      """Initializes DummyChild."""
      self.leading_trivia = ["trivia"]

  class DummyNode:
    """A dummy node."""

    def __init__(self):
      """Initializes DummyNode."""
      self.children = [DummyChild()]

  assert _get_trivia(DummyNode()) == ["trivia"]

  # test directive param list fallback (line 240, 242)
  # Not sure exactly how to hit it from parser, so we'll mock or force it via parser
  from lark import Tree, Token
  from ml_switcheroo.core.compiler.frontends.sass.parser import SassTransformer

  transformer = SassTransformer()

  # line 242 (param_list is not list)
  d = transformer.directive([Token("DOT", "."), Token("IDENTIFIER", "req"), "string_param"])
  assert d.params == ["string_param"]

  # line 240 (param inside list is not Token or at_string or list)
  d = transformer.directive([Token("DOT", "."), Token("IDENTIFIER", "req"), [Token("IDENTIFIER", "abc"), 123]])
  assert d.params == ["abc", "123"]

  # line 234 (at_string tree)
  at_str = Tree("at_string", [Token("AT", "@"), Token("STRING", '"val"')])
  d = transformer.directive([Token("DOT", "."), Token("IDENTIFIER", "req"), [at_str]])
  assert d.params == ['@"val"']


def test_sass_parser_branch_coverage():
  """Test for test_sass_parser_branch_coverage."""
  from ml_switcheroo.core.compiler.frontends.sass.parser import SassTransformer

  transformer = SassTransformer()
  # instruction with all None children (300->305)
  i = transformer.instruction([None, None])
  assert i.opcode == ""
