"""SDV tabular synthesizer adapter (CTGAN / TVAE / CopulaGAN)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

from buildml.core.errors import ValidationError
from buildml.synthetic.extras import require_sdv
from buildml.synthetic.types import ColumnSchemaSpec

SdvMethod = Literal["ctgan", "tvae", "copulagan"]


def _build_synthesizer(
    method: SdvMethod,
    metadata: Any,
    *,
    epochs: int,
    batch_size: int,
    verbose: bool = False,
) -> Any:
    require_sdv()
    if method == "ctgan":
        from sdv.single_table import CTGANSynthesizer

        return CTGANSynthesizer(
            metadata,
            epochs=int(epochs),
            batch_size=int(batch_size),
            verbose=verbose,
        )
    if method == "tvae":
        from sdv.single_table import TVAESynthesizer

        return TVAESynthesizer(
            metadata,
            epochs=int(epochs),
            batch_size=int(batch_size),
            verbose=verbose,
        )
    if method == "copulagan":
        from sdv.single_table import CopulaGANSynthesizer

        return CopulaGANSynthesizer(
            metadata,
            epochs=int(epochs),
            batch_size=int(batch_size),
            verbose=verbose,
        )
    raise ValidationError(f"Unsupported SDV synthesizer method: {method!r}")


@dataclass
class SdvTabularGenerator:
    """Train-fitted SDV single-table synthesizer wrapper."""

    method: SdvMethod
    columns: tuple[str, ...]
    synthesizer: Any = field(default=None, repr=False)
    metadata: Any = field(default=None, repr=False)
    column_specs: tuple[ColumnSchemaSpec, ...] = ()
    epochs: int = 300
    batch_size: int = 500
    random_state: int = 42
    n_rows_fitted: int = 0

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        specs: tuple[ColumnSchemaSpec, ...],
        *,
        method: SdvMethod,
        epochs: int = 300,
        batch_size: int = 500,
        random_state: int = 42,
        verbose: bool = False,
    ) -> SdvTabularGenerator:
        """Run fit on input data using the fitted internal state.

Called from the Session-facing workflow after splits and roles are set. Validation and test partitions are evaluation-only unless explicitly documented.

Parameters
----------
frame:
    Partition or full DataFrame slice used for this operation.
specs:
    specs (tuple[ColumnSchemaSpec, ...]).
method:
    Method or strategy identifier for the resolved backend.
epochs:
    Training epochs for torch-backed estimators.
batch_size:
    Number of rows to select per query or training minibatch.
random_state:
    Seed for stochastic steps (sampling, initialization, bagging).
verbose:
    verbose (bool).

Returns
-------
SdvTabularGenerator
    Return value (SdvTabularGenerator) produced by this operation.

Raises
------
ValidationError
    When preconditions for this operation are not met.
        """
        require_sdv(feature=f"SDV method='{method}'")
        if len(frame) < 10:
            raise ValidationError(
                "SDV synthesizers need at least 10 train rows for stable fit."
            )
        from sdv.metadata import SingleTableMetadata

        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(frame)
        synthesizer = _build_synthesizer(
            method,
            metadata,
            epochs=int(epochs),
            batch_size=int(batch_size),
            verbose=verbose,
        )
        synthesizer.fit(frame.reset_index(drop=True))
        return cls(
            method=method,
            columns=tuple(frame.columns),
            synthesizer=synthesizer,
            metadata=metadata,
            column_specs=specs,
            epochs=int(epochs),
            batch_size=int(batch_size),
            random_state=int(random_state),
            n_rows_fitted=int(len(frame)),
        )

    def sample(
        self,
        n: int,
        *,
        random_state: int | None = None,
        condition: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Run sample on input data using the fitted internal state.

Called from the Session-facing workflow after splits and roles are set. Validation and test partitions are evaluation-only unless explicitly documented.

Parameters
----------
n:
    n (int).
random_state:
    Seed for stochastic steps (sampling, initialization, bagging).
condition:
    condition (dict[str, Any] | None).

Returns
-------
pd.DataFrame
    Return value (pd.DataFrame) produced by this operation.

Raises
------
ValidationError
    When preconditions for this operation are not met.
        """
        if self.synthesizer is None:
            raise ValidationError("SdvTabularGenerator is not fitted.")
        if condition:
            raise ValidationError(
                "condition= is not supported for SDV synthesizers in this path. "
                "Use native method='gaussian_copula' for rejection sampling."
            )
        _ = random_state if random_state is not None else self.random_state
        out = self.synthesizer.sample(num_rows=int(n))
        if not isinstance(out, pd.DataFrame):
            out = pd.DataFrame(out)
        return out[list(self.columns)].reset_index(drop=True)
