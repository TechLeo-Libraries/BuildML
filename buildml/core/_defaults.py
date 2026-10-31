"""Internal defaults that distinguish omission from an explicit permission."""
from typing import cast


class _Unspecified:
    def __repr__(self) -> str:
        return "UNSET"


# Retain bool annotations and reject explicit nonbooleans, including None.
_INHERIT_PERMISSION = cast(bool, _Unspecified())
