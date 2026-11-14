"""Plugin for Normalizing Tensor Padding Coordinate Formats.

Normalizes flat reverse-pair padding tuples (4D NCHW and 5D NCDHW) and
already-nested per-dimension pair tuples across positional and keyword call
forms using ``PluginTraits.has_numpy_compatible_arrays`` and
``HookContext.current_variant``.
"""

import typing

import libcst as cst

from ml_switcheroo.core.hooks import HookContext, register_hook
from ml_switcheroo.plugins.utils import create_dotted_name, resolve_variant_api
from ml_switcheroo.semantics.resolution import query_framework_traits

_INPUT_KEYWORDS = frozenset({"input", "x", "array", "tensor"})
_PAD_KEYWORDS = frozenset({"pad", "pad_width", "paddings"})


def _create_zero_pad() -> cst.Element:
  """Create a ``(0, 0)`` padding pair tuple element.

  Returns:
      CST ``Element`` representing ``(0, 0)``.
  """
  return cst.Element(
    value=cst.Tuple(
      elements=[
        cst.Element(cst.Integer("0"), comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" "))),
        cst.Element(cst.Integer("0")),
      ]
    )
  )


def _create_dim_pad(before: cst.BaseExpression, after: cst.BaseExpression) -> cst.Element:
  """Create a ``(before, after)`` padding pair tuple element.

  Args:
      before: Padding expression before the dimension.
      after: Padding expression after the dimension.

  Returns:
      CST ``Element`` representing ``(before, after)``.
  """
  return cst.Element(
    value=cst.Tuple(
      elements=[
        cst.Element(before, comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" "))),
        cst.Element(after),
      ]
    )
  )


def _supports_numpy_padding(ctx: HookContext) -> bool:
  """Check if the target framework supports tuple-of-tuples padding via ``PluginTraits``.

  Args:
      ctx: Hook context containing semantics and target framework metadata.

  Returns:
      True if the target framework supports NumPy-style tuple-of-tuples padding.
  """
  if ctx.semantics is None:
    return False

  fw_traits = query_framework_traits(ctx.semantics, ctx.target_framework)
  return bool(fw_traits.plugin_traits is not None and fw_traits.plugin_traits.has_numpy_compatible_arrays)


def _extract_input_and_pad_indices(args: typing.Sequence[cst.Arg]) -> tuple[int, int] | None:
  """Locate the indices of the input tensor and padding tuple arguments.

  Args:
      args: Sequence of call arguments.

  Returns:
      ``(input_idx, pad_idx)`` if both arguments are present, otherwise ``None``.
  """
  input_idx: int | None = None
  pad_idx: int | None = None
  pos_indices: list[int] = []

  for idx, arg in enumerate(args):
    if arg.star:
      continue
    if arg.keyword is not None:
      kw = arg.keyword.value
      if kw in _INPUT_KEYWORDS:
        input_idx = idx
      elif kw in _PAD_KEYWORDS:
        pad_idx = idx
    else:
      pos_indices.append(idx)

  if input_idx is None and pos_indices:
    input_idx = pos_indices.pop(0)
  if pad_idx is None and pos_indices:
    pad_idx = pos_indices.pop(0)

  if input_idx is None or pad_idx is None:
    return None
  return input_idx, pad_idx


def _convert_pad_tuple(pad_tuple: cst.Tuple) -> cst.Tuple | None:
  """Convert flat reverse-pair tuples or validate pre-nested pair tuples.

  Args:
      pad_tuple: Source padding ``cst.Tuple`` node.

  Returns:
      Normalized tuple-of-pairs ``cst.Tuple``, or ``None`` if unsupported shape.
  """
  elements = pad_tuple.elements
  if elements and all(isinstance(el.value, (cst.Tuple, cst.List)) and len(el.value.elements) == 2 for el in elements):
    return pad_tuple

  if len(elements) == 4:
    left, right, top, bottom = (el.value for el in elements)
    return cst.Tuple(
      elements=[
        _create_zero_pad(),
        _create_zero_pad(),
        _create_dim_pad(top, bottom),
        _create_dim_pad(left, right),
      ]
    )

  if len(elements) == 6:
    left, right, top, bottom, front, back = (el.value for el in elements)
    return cst.Tuple(
      elements=[
        _create_zero_pad(),
        _create_zero_pad(),
        _create_dim_pad(front, back),
        _create_dim_pad(top, bottom),
        _create_dim_pad(left, right),
      ]
    )

  return None


@register_hook("padding_converter")
def transform_padding(node: cst.Call, ctx: HookContext) -> cst.Call:
  """Transform padding coordinate format into NumPy-compatible tuple-of-tuples.

  Args:
      node: Original padding ``cst.Call`` node.
      ctx: Hook context containing target framework traits and variant metadata.

  Returns:
      Transformed ``cst.Call`` node if supported and mapped, else ``node``.
  """
  if not _supports_numpy_padding(ctx):
    return node

  target_api = resolve_variant_api(ctx, ("Pad", "pad"))
  if not target_api:
    return node

  args = list(node.args)
  located = _extract_input_and_pad_indices(args)
  if located is None:
    return node

  input_idx, pad_idx = located
  input_arg = args[input_idx]
  pad_arg = args[pad_idx]

  if not isinstance(pad_arg.value, cst.Tuple):
    return node

  new_pad_tuple = _convert_pad_tuple(pad_arg.value)
  if new_pad_tuple is None:
    return node

  clean_input = input_arg.with_changes(
    keyword=None,
    equal=cst.MaybeSentinel.DEFAULT,
    comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")),
  )
  clean_pad = pad_arg.with_changes(
    keyword=None,
    equal=cst.MaybeSentinel.DEFAULT,
    value=new_pad_tuple,
    comma=cst.MaybeSentinel.DEFAULT,
  )
  return node.with_changes(
    func=create_dotted_name(target_api),
    args=[clean_input, clean_pad],
  )
