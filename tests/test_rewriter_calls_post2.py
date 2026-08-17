"""Module docstring."""

import libcst as cst
from ml_switcheroo.core.rewriter.calls.post import handle_post_processing


class DummyTraits:
  """Docstring."""

  def __init__(self):
    """Docstring."""
    self.strip_magic_args = []
    self.auto_strip_magic_args = False
    self.inject_magic_args = []


class DummyContext:
  """Docstring."""

  def __init__(self, is_init=False, is_module=False):
    """Docstring."""

    class Sig:
      def __init__(self, i, m):
        self.is_init = i
        self.is_module_method = m

    self.signature_stack = [Sig(is_init, is_module)] if is_init or is_module else []


class DummySemantics:
  """Docstring."""

  def __init__(self, origins=None, magic=None):
    """Docstring."""
    self._key_origins = origins or {}
    if magic is not None:
      self.known_magic_args = magic


class DummyRewriter:
  """Docstring."""

  def __init__(self, context=None, semantics=None, traits=None, report=True):
    """Docstring."""
    self.context = context or DummyContext()
    self.semantics = semantics or DummySemantics()
    self.traits = traits or DummyTraits()
    self.report = report

  def _create_dotted_name(self, name):
    if name == "error":
      raise ValueError("bad type")
    return cst.Name(name)

  def _report_failure(self, msg):
    pass

  def _get_target_traits(self):
    return self.traits


def test_post_branches():
  """Docstring."""
  rewriter = DummyRewriter()
  node = cst.parse_statement("f()").body[0].value

  # 38 -> 47 (missing output_select_index)
  handle_post_processing(rewriter, node, {}, "id")

  # 47 -> 59 (missing output_cast)
  handle_post_processing(rewriter, node, {"output_select_index": None}, "id")

  # 38 -> 39, 47 -> 48 -> 59
  mapping = {"output_select_index": 0, "output_cast": "float32"}
  node_tuple = cst.parse_statement("f()[0]").body[0].value
  handle_post_processing(rewriter, node_tuple, mapping, "id")

  # 59 -> 99 (no context signature stack)
  handle_post_processing(rewriter, node, mapping, "id")
