"""Module docstring."""

import libcst as cst
from ml_switcheroo.core.mlir.type_inference import parse_py_type_to_mlir, TypeInferencePass
from ml_switcheroo.core.mlir.types import FloatType


def test_parse_py_type_to_mlir() -> None:
  """Docstring."""
  assert parse_py_type_to_mlir("int").to_string() == "i32"
  assert parse_py_type_to_mlir("float").to_string() == "f32"
  assert parse_py_type_to_mlir("bool").to_string() == "i1"
  assert parse_py_type_to_mlir("tensor").to_string() == "tensor<*xf32>"
  assert parse_py_type_to_mlir("array").to_string() == "tensor<*xf32>"
  assert parse_py_type_to_mlir("unknown").to_string() == "!sw.unknown"


def test_type_inference_pass_branches() -> None:
  """Docstring."""
  # 67 -> 68, 69 -> 70
  code: str = """
def func(x: float, y: int):
    return x
"""
  mod: cst.Module = cst.parse_module(code)
  infer: TypeInferencePass = TypeInferencePass(initial_env={"x": parse_py_type_to_mlir("float")})
  mod.visit(infer)
  assert len(infer.return_types) == 1

  # 93 -> 94
  code2: str = """
def f(x):
    if x:
        return 1
    else:
        return 2
"""
  mod2: cst.Module = cst.parse_module(code2)
  infer2: TypeInferencePass = TypeInferencePass(initial_env={})
  mod2.visit(infer2)

  # 96 -> 97, 98 -> 99
  code3: str = """
def g(x):
    return
"""
  mod3: cst.Module = cst.parse_module(code3)
  infer3: TypeInferencePass = TypeInferencePass(initial_env={})
  mod3.visit(infer3)

  # Missing assignments to hit 78 -> 82 and 78 -> 79
  code4: str = """
x = 1
"""
  mod4: cst.Module = cst.parse_module(code4)
  infer4: TypeInferencePass = TypeInferencePass(initial_env={})
  mod4.visit(infer4)

  # Let's test assign where we DO have the env defined
  code5: str = """
y = x
"""
  mod5: cst.Module = cst.parse_module(code5)
  infer5: TypeInferencePass = TypeInferencePass(initial_env={"x": FloatType("f32")})
  mod5.visit(infer5)

  # Return multiple
  code6: str = """
def h(x):
    return 1, 2
"""
  mod6: cst.Module = cst.parse_module(code6)
  infer6: TypeInferencePass = TypeInferencePass(initial_env={})
  mod6.visit(infer6)


def test_type_inference_assign_not_name() -> None:
  """Function doc."""
  code: str = """
obj.attr = 1
"""
  mod: cst.Module = cst.parse_module(code)
  infer: TypeInferencePass = TypeInferencePass(initial_env={})
  mod.visit(infer)


def test_type_inference_float_assign() -> None:
  """Function doc."""
  code: str = """
x = 1.0
"""
  mod: cst.Module = cst.parse_module(code)
  infer: TypeInferencePass = TypeInferencePass(initial_env={})
  mod.visit(infer)
  assert infer.env["x"].to_string() == "f32"
