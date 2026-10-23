"""Validation models."""

from ml_switcheroo.enums import RuntimeTensorOrValue
import typing

from pydantic import JsonValue


class SemanticsValidator:
  """Schema validation for Operation definitions."""

  @staticmethod
  def validate_and_dump(
    abstract_id: str, new_data: typing.Mapping[str, RuntimeTensorOrValue]
  ) -> dict[str, JsonValue] | None:
    """Validate against the OperationDef model and return dumped data or None on failure.

    Args:
        abstract_id: Canonical abstract operation identifier.
        new_data: Raw operation definition dictionary to validate.

    Returns:
        Validated and serialized dictionary if valid, else None.
    """
    from pydantic import ValidationError

    from ml_switcheroo.core.dsl import OperationDef

    details_to_validate = dict(new_data)

    # 1. Inject missing fields required by Schema if not present
    if "operation" not in details_to_validate:
      details_to_validate["operation"] = abstract_id
    if "variants" not in details_to_validate:
      details_to_validate["variants"] = {}
    if "description" not in details_to_validate:
      details_to_validate["description"] = f"Definition for {abstract_id}"
    if "std_args" not in details_to_validate:
      details_to_validate["std_args"] = []

    try:
      validated = OperationDef.model_validate(details_to_validate)
      res: dict[str, JsonValue] = validated.model_dump(by_alias=True, exclude_unset=True)
      return res
    except ValidationError as e:
      print(f"❌ Cannot update invalid definition for '{abstract_id}': {e}")
      return None
