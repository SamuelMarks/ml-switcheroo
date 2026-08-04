"""Tests for Ghost Core (Introspection & Serialization).

Verifies that:
1. Live objects (Functions) are correctly inspected.
2. Live objects (Classes) have their __init__ signature extracted.
3. Serialization/Hydration (Ghost Mode) works seamlessly.
4. Edge cases (defaults, annotations, missing signatures) are handled.
"""

from typing import Optional
from ml_switcheroo.core.ghost import GhostInspector


# --- Mock Objects for Inspection ---


def simple_func(x, y=10):
  """A simple mock function used to test basic inspection of parameters and defaults.

  Args:
      x: The first operand, typically a number.
      y: The second operand, typically a number with a default of 10.

  Returns:
      The sum of x and y.
  """
  return x + y


def typed_func(x: int, opt: Optional[str] = None):
  """A mock function with type annotations to test type hint stringification.

  Args:
      x: An integer argument to test standard type extraction.
      opt: An optional string argument to test Union/Optional type extraction.

  Returns:
      None.
  """
  pass


class SimpleClass:
  """A standard mock class to test introspection of class __init__ methods.

  This class simulates a typical neural network layer or model block with an
  initialization method containing multiple parameters.
  """

  def __init__(self, output_dim, activation="relu"):
    """Initializes the SimpleClass instance.

    Args:
        output_dim: The output dimension of the simulated layer.
        activation: The activation function name, defaulting to "relu".

    Returns:
        None.
    """
    self.out = output_dim


class BuiltinLike:
  """A class mimicking built-ins that might fail signature inspection.

  Used to test the fallback behavior of GhostInspector when standard signature
  inspection raises a ValueError.
  """

  pass


# --- Tests ---


def test_inspect_simple_function():
  """Verifies that basic function inspection extracts correct parameters, defaults, and attributes.

  Returns:
      None.
  """
  ref = GhostInspector.inspect(simple_func, "test.simple_func")

  assert ref.name == "simple_func"
  assert ref.kind == "function"
  assert ref.api_path == "test.simple_func"
  assert (
    ref.docstring
    == "A simple mock function used to test basic inspection of parameters and defaults.\n\nArgs:\n    x: The first operand, typically a number.\n    y: The second operand, typically a number with a default of 10.\n\nReturns:\n    The sum of x and y."
  )

  assert len(ref.params) == 2
  p0, p1 = ref.params

  assert p0.name == "x"
  assert p0.default is None

  assert p1.name == "y"
  assert p1.default == "10"


def test_inspect_class_init():
  """Verifies that class inspection inspects the __init__ signature and skips 'self'.

  Returns:
      None.
  """
  ref = GhostInspector.inspect(SimpleClass, "test.SimpleClass")

  assert ref.name == "SimpleClass"
  assert ref.kind == "class"
  assert (
    ref.docstring
    == "A standard mock class to test introspection of class __init__ methods.\n\nThis class simulates a typical neural network layer or model block with an\ninitialization method containing multiple parameters."
  )

  # Should detect 'output_dim' and 'activation'
  # Should SKIP 'self'
  assert len(ref.params) == 2
  assert ref.params[0].name == "output_dim"
  assert ref.params[1].name == "activation"
  assert ref.params[1].default == "relu"


def test_inspect_typed_signature():
  """Verifies that type hints are successfully extracted and stringified by the inspector.

  Returns:
      None.
  """
  ref = GhostInspector.inspect(typed_func, "test.typed")

  p0 = ref.params[0]
  assert p0.name == "x"
  assert p0.annotation == "int"

  p1 = ref.params[1]
  assert p1.name == "opt"
  # Representation of Optional[...] varies by python version, just check it exists stringified
  assert p1.annotation is not None
  assert "Optional" in p1.annotation or "Union" in p1.annotation or "None" in p1.annotation


def test_ghost_hydration_roundtrip():
  """Verifies that GhostRef can be serialized to a dict and hydrated back correctly.

  Returns:
      None.
  """
  # 1. Inspect Live
  live_ref = GhostInspector.inspect(simple_func, "func")

  # 2. Serialize
  data = live_ref.model_dump()

  # 3. Hydrate
  ghost_ref = GhostInspector.hydrate(data)

  assert ghost_ref == live_ref
  assert ghost_ref.has_arg("y")
  assert not ghost_ref.has_arg("z")


def test_inspect_failure_handling(monkeypatch):
  """Verifies that the inspector falls back safely when signature inspection raises ValueError.

  Args:
      monkeypatch: The pytest monkeypatch fixture used to mock inspect.signature.

  Returns:
      None.
  """

  # We patch signature to raise ValueError mimics C-ext failure
  def mock_sig(obj):
    """Mocks inspect.signature to simulate inspection failures.

    Args:
        obj: The object whose signature is being requested.

    Returns:
        Never returns normally; always raises ValueError.

    Raises:
        ValueError: Always raised to simulate inspection failures on C-extensions.
    """
    raise ValueError("no signature found")

  monkeypatch.setattr("inspect.signature", mock_sig)

  ref = GhostInspector.inspect(BuiltinLike, "test.BuiltinLike")

  # Should return a valid Ref object with empty params
  assert ref.name == "BuiltinLike"
  assert ref.params == []
  # Docstring should still be captured if inspect.getdoc works
  assert "mimicking" in (ref.docstring or "")


def test_ghost_ref_helper_methods():
  """Verifies helper methods such as has_arg on the GhostRef class.

  Returns:
      None.
  """
  ref = GhostInspector.inspect(simple_func, "foo")

  assert ref.has_arg("x") is True
  assert ref.has_arg("non_existent") is False


def test_ghost_varargs():
  """Verifies that GhostInspector correctly identifies functions with variable positional arguments (*args).

  Returns:
      None.
  """

  def func_with_args(*args):
    """A mock function accepting variable positional arguments.

    Args:
        *args: Variable length argument list.

    Returns:
        None.
    """
    pass

  ref = GhostInspector.inspect(func_with_args, "func_with_args")
  assert ref.has_varargs is True


def test_ghost_callable_default():
  """Verifies that callables used as default values are treated as None to avoid serializing memory addresses.

  Returns:
      None.
  """

  def some_callable():
    """A mock callable default value.

    Returns:
        None.
    """
    pass

  def func_with_callable(cb=some_callable):
    """A mock function with a callable as a default argument.

    Args:
        cb: A callback function, defaulting to some_callable.

    Returns:
        None.
    """
    pass

  ref = GhostInspector.inspect(func_with_callable, "func_with_callable")
  assert ref.params[0].default is None


def test_ghost_unrepresentable_default():
  """Verifies handling of default arguments with unrepresentable string values.

  Returns:
      None.
  """

  class BadRepr:
    """A class designed to have a string representation that includes a hex address.

    Used to test that standard string representations containing addresses are ignored
    or converted to None.
    """

    def __repr__(self):
      """Returns a safe representation of BadRepr.

      Returns:
          The string "GoodRepr".
      """
      return "GoodRepr"

    def __str__(self):
      """Returns a string representation that mimics a memory address.

      Returns:
          A string containing a mock memory address.
      """
      return "BadStr at 0x123"

  class ExplodingStr:
    """A class designed to raise an exception when stringified.

    Used to test robustness of default value string conversion.
    """

    def __repr__(self):
      """Returns a safe representation of ExplodingStr.

      Returns:
          The string "SafeRepr".
      """
      return "SafeRepr"

    def __str__(self):
      """Simulates a failure during string conversion.

      Returns:
          Never returns normally.

      Raises:
          Exception: Always raised to simulate a failure.
      """
      raise Exception("Boom")

  def func_with_bad_str(x=BadRepr(), y=ExplodingStr()):
    """A mock function with unrepresentable default values.

    Args:
        x: A parameter with a mock address string representation, defaulting to BadRepr().
        y: A parameter that raises an exception on __str__, defaulting to ExplodingStr().

    Returns:
        None.
    """
    pass

  ref = GhostInspector.inspect(func_with_bad_str, "func_with_bad_str")
  assert ref.params[0].default is None
  assert ref.params[1].default == "<unrepresentable>"


def test_ghost_annotation_without_name():
  """Verifies that annotations without a standard name attribute are serialized correctly using their string representation.

  Returns:
      None.
  """

  class NoNameAnnotation:
    """An annotation mock object that does not possess a __name__ attribute.

    Used to test fallback logic in type annotation stringification.
    """

    def __str__(self):
      """Returns the string representation of the annotation object.

      Returns:
          The string "NoName".
      """
      return "NoName"

  def func_with_anno(x: NoNameAnnotation()):
    """A mock function that uses a custom instance as a type annotation.

    Args:
        x: A parameter annotated with an instance of NoNameAnnotation.

    Returns:
        None.
    """
    pass

  ref = GhostInspector.inspect(func_with_anno, "func_with_anno")
  assert ref.params[0].annotation == "NoName"


def test_ghost_function_c_extension_fallback(monkeypatch):
  """Verifies fallback parameter generation for regular functions when inspection raises ValueError (e.g. C-extensions).

  Args:
      monkeypatch: The pytest monkeypatch fixture used to mock inspect.signature.

  Returns:
      None.
  """

  def mock_sig(obj):
    """Mocks inspect.signature to raise ValueError.

    Args:
        obj: The object whose signature is being inspected.

    Returns:
        Never returns normally; always raises ValueError.

    Raises:
        ValueError: Always raised to simulate inspection failure.
    """
    raise ValueError("no signature found")

  monkeypatch.setattr("inspect.signature", mock_sig)

  def dummy_func():
    """A mock dummy function to test fallback behavior.

    Returns:
        None.
    """
    pass

  ref = GhostInspector.inspect(dummy_func, "dummy_func")
  assert ref.name == "dummy_func"
  assert ref.has_varargs is True
  assert len(ref.params) == 2
  assert ref.params[0].name == "args"
  assert ref.params[1].name == "kwargs"
