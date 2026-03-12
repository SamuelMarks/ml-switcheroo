def test_generator_empty_semantics():
  from ml_switcheroo.generated_tests.generator import TestCaseGenerator
  from pathlib import Path

  gen = TestCaseGenerator(None)
  gen.generate({}, Path("foo.py"))


def test_templates_exception():
  from ml_switcheroo.generated_tests.templates import get_template

  class FaultyManager:
    def get_test_template(self, fw):
      raise ValueError("fail")

  assert get_template(FaultyManager(), "torch") != {}


def test_inputs_parse_arg_def():
  from ml_switcheroo.generated_tests.inputs import parse_arg_def

  assert parse_arg_def({"name": "foo", "type": "Any", "default": True})["type"] == "bool"
  assert parse_arg_def({"name": "foo", "type": "Any", "default": 1})["type"] == "int"
  assert parse_arg_def({"name": "foo", "type": "Any", "default": 1.0})["type"] == "float"
  assert parse_arg_def({"name": "foo", "type": "Any", "default": []})["type"] == "Array"
  assert parse_arg_def("just_a_string") == {"name": "just_a_string", "type": "Array"}


def test_infer_type_from_default():
  from ml_switcheroo.generated_tests.inputs import _infer_type_from_default

  assert _infer_type_from_default(True) == "bool"
  assert _infer_type_from_default(1) == "int"
  assert _infer_type_from_default(1.0) == "float"
  assert _infer_type_from_default([1, 2]) == "List[int]"
  assert _infer_type_from_default([1.0, 2.0]) == "List[Any]"


def test_inputs_generate_input_value_code():
  from ml_switcheroo.generated_tests.inputs import generate_input_value_code

  assert "random.choice([1, 2])" in generate_input_value_code("foo", {"options": [1, 2]})

  assert "random.randint" in generate_input_value_code("foo", {"type": "int", "min": 5, "max": 10})
  assert "random.randint" in generate_input_value_code("foo", {"type": "int"})
  assert "random.randint(5," in generate_input_value_code("foo", {"type": "int", "min": 5})
  assert "random.randint(" in generate_input_value_code("foo", {"type": "int", "max": 10})

  assert "random.uniform" in generate_input_value_code("foo", {"type": "float", "min": 0.5, "max": 1.5})
  assert "random.uniform(0.5," in generate_input_value_code("foo", {"type": "float", "min": 0.5})
  assert "random.uniform(" in generate_input_value_code("foo", {"type": "float", "max": 1.5})

  assert "[1, 2]" == generate_input_value_code("foo", {"type": "List[int]", "default": [1, 2]})
  assert "(1, 2)" == generate_input_value_code("foo", {"type": "Tuple[int]", "default": (1, 2)})

  assert "bool(random.getrandbits(1))" in generate_input_value_code("foo", "bool")

  assert "np.random.uniform" in generate_input_value_code("foo", {"type": "Array", "dtype": "int", "min": 5, "max": 10})
  assert "np.random.randn" in generate_input_value_code("foo", {"type": "Array", "dtype": "bool"})

  assert "bool(random.getrandbits(1))" in generate_input_value_code("foo", {"type": "Any", "default": True})

  # 126-133: explicit bounds without type
  assert "random.randint" in generate_input_value_code("foo", {"type": "Any", "min": 1, "max": 2})
  assert "random.uniform" in generate_input_value_code("foo", {"type": "Any", "min": 1.0, "max": 2.0})
  assert "None" in generate_input_value_code("foo", {"type": "Callable"})

  assert "[1, 2]" == generate_input_value_code("foo", {"type": "List[int]"})
  assert "(1, 2)" == generate_input_value_code("foo", {"type": "Tuple[int]"})


def test_generate_dim_heuristic():
  from ml_switcheroo.generated_tests.inputs import _generate_dim_heuristic

  assert _generate_dim_heuristic("axis") == "1"
  assert _generate_dim_heuristic("keepdims") == "bool(random.getrandbits(1))"
  assert _generate_dim_heuristic("foo") == "1"


def test_inputs_parse_arg_def_more():
  from ml_switcheroo.generated_tests.inputs import parse_arg_def

  assert parse_arg_def(("foo", "int")) == {"name": "foo", "type": "int"}
  assert parse_arg_def({"name": "foo", "type": "Any", "default": object()})["type"] == "Array"
  assert parse_arg_def({"name": "foo", "type": "Any"})["type"] == "Array"
  assert parse_arg_def(123) == {"name": "unknown", "type": "Array"}


def test_infer_type_from_default_more():
  from ml_switcheroo.generated_tests.inputs import _infer_type_from_default

  assert _infer_type_from_default(["abc"]) == "List[Any]"


def test_inputs_generate_input_value_code_more():
  from ml_switcheroo.generated_tests.inputs import generate_input_value_code

  assert "random.randint" in generate_input_value_code("foo", {"type": "Any", "min": 1})
  assert "random.randint" in generate_input_value_code("foo", {"type": "Any", "max": 10})
  assert "+ 1" in generate_input_value_code("foo", {"type": "Array", "min": 1})
  assert "astype(np.float32)" in generate_input_value_code("foo", {"type": "Array"})


def test_inputs_infer_type_from_default_any():
  from ml_switcheroo.generated_tests.inputs import _infer_type_from_default

  assert _infer_type_from_default("abc") == "Any"


def test_inputs_generate_dim_heuristic_fallback():
  from ml_switcheroo.generated_tests.inputs import generate_input_value_code

  assert "1" in generate_input_value_code("axis", {"type": "Any"})


def test_inputs_generate_dim_heuristic_fallback_2():
  from ml_switcheroo.generated_tests.inputs import generate_input_value_code

  assert "np.random.randn" in generate_input_value_code("foo_bar", {"type": "Any"})


def test_templates_is_static_arg():
  from ml_switcheroo.generated_tests.templates import is_static_arg, get_template

  assert is_static_arg({"type": "int"}) is True
  assert is_static_arg({"type": "bool"}) is True
  assert is_static_arg({"type": "str"}) is True
  assert is_static_arg({"type": "list[int]"}) is True
  assert is_static_arg({"type": "tuple[int]"}) is True

  assert is_static_arg({"type": "Array", "name": "axis"}) is True
  assert is_static_arg({"type": "Array", "name": "dim"}) is True
  assert is_static_arg({"type": "Array", "name": "keepdims"}) is True

  assert is_static_arg({"type": "Array", "name": "foo"}) is False

  class GoodManager:
    def get_test_template(self, fw):
      return {"import": "foo"}

  assert get_template(GoodManager(), "jax") == {"import": "foo"}
