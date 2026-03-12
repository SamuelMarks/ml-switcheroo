import pytest
import ast
from pathlib import Path
from unittest.mock import patch, mock_open
from ml_switcheroo.discovery.harvester import ImportScanner, SemanticHarvester
from ml_switcheroo.semantics.manager import SemanticsManager


def test_import_scanner_from():
  scanner = ImportScanner("jax")
  code = "from jax.numpy import add as jnp_add, sub"
  tree = ast.parse(code)
  scanner.visit(tree)
  assert scanner.aliases["jnp_add"] == "jax.numpy.add"
  assert scanner.aliases["sub"] == "jax.numpy.sub"


def test_harvester_file_not_found():
  harv = SemanticHarvester(SemanticsManager(), "jax")
  with patch("pathlib.Path.exists", return_value=False):
    assert harv.harvest_file(Path("fake.py")) == 0


def test_harvester_parse_error():
  harv = SemanticHarvester(SemanticsManager(), "jax")
  with patch("pathlib.Path.exists", return_value=True), patch("pathlib.Path.read_text", side_effect=Exception("error")):
    assert harv.harvest_file(Path("fake.py")) == 0


def test_harvester_invalid_test_name():
  harv = SemanticHarvester(SemanticsManager(), "jax")
  with (
    patch("pathlib.Path.exists", return_value=True),
    patch("pathlib.Path.read_text", return_value="def test_gen_(): pass"),
  ):
    # _infer_op_name will return None
    assert harv.harvest_file(Path("fake.py")) == 0


def test_infer_op_name():
  harv = SemanticHarvester(SemanticsManager(), "jax")
  assert harv._infer_op_name("test_gen_abs") == "abs"
  assert harv._infer_op_name("test_abs") == "abs"
  assert harv._infer_op_name("foo") is None


def test_analyze_test_body_missing_api():
  harv = SemanticHarvester(SemanticsManager(), "jax")
  # 185
  with patch.object(harv.semantics, "get_definition_by_id", return_value={"variants": {"jax": {}}}):
    assert harv._analyze_test_body(None, "op", {}) is None
  # 180
  with patch.object(harv.semantics, "get_definition_by_id", return_value=None):
    assert harv._analyze_test_body(None, "op", {}) is None


def test_apply_update_edge_cases():
  harv = SemanticHarvester(SemanticsManager(), "jax")

  # 209: None
  with patch.object(harv.semantics, "get_definition_by_id", return_value=None):
    assert harv._apply_update("op", {}, False) is False

  # 215: target_fw not in variants
  with patch.object(harv.semantics, "get_definition_by_id", return_value={"variants": {"torch": {}}}):
    assert harv._apply_update("op", {}, False) is False

  # 219: target_variant not dict
  with patch.object(harv.semantics, "get_definition_by_id", return_value={"variants": {"jax": "string"}}):
    assert harv._apply_update("op", {}, False) is False

  # 224: old_args == arg_map
  with patch.object(harv.semantics, "get_definition_by_id", return_value={"variants": {"jax": {"args": {"a": "b"}}}}):
    assert harv._apply_update("op", {"a": "b"}, False) is False


def test_call_inspector_normalize_info():
  from ml_switcheroo.discovery.harvester import TargetCallVisitor

  insp = TargetCallVisitor("tgt", [], {})

  # 280-287
  res = insp._normalize_info(
    [
      {"name": "x", "type": "int"},
      {"type": "float"},  # no name
      "y",
    ]
  )
  assert res == [("x", "int"), ("y", "Any")]


def test_call_inspector_visit_call():
  from ml_switcheroo.discovery.harvester import TargetCallVisitor
  import ast

  insp = TargetCallVisitor("tgt", [], {})

  # 298
  insp.mappings = {}
  insp.visit_Call(ast.Call(func=ast.Name(id="f")))

  # 302
  insp.mappings = None
  with patch.object(insp, "_resolve_call_name", return_value=None):
    insp.visit_Call(ast.Call(func=ast.Name(id="f")))

  # 316 kw arg is None (e.g. **kwargs)
  with patch.object(insp, "_resolve_call_name", return_value="tgt"):
    node = ast.Call(func=ast.Name(id="tgt"), args=[], keywords=[ast.keyword(arg=None, value=ast.Name(id="kwargs"))])
    insp.visit_Call(node)


def test_harvester_call_inspector_helpers():
  from ml_switcheroo.discovery.harvester import TargetCallVisitor
  import ast

  insp = TargetCallVisitor("tgt", [], {})

  # 367: resolve call name non-attribute non-name
  assert insp._resolve_call_name(ast.Constant(value=1)) == ""

  # 395: clean std name not np_
  assert insp._clean_std_name("var") == "var"

  # 427: infer literal type unknown
  assert insp._infer_literal_type(ast.Call(func=ast.Name(id="f"))) == "Any"

  # 455: infer container type Any
  assert insp._infer_container_type([ast.Call(func=ast.Name(id="f"))], "List") == "List"

  # 509: Priority ambiguity fallback
  insp.std_args_info = [("axis", "int"), ("dim", "int"), ("other", "int")]
  # passing val_type "int" yields 3 candidates. priority finds "axis".
  assert insp._find_std_arg_by_type("int") == "axis"
