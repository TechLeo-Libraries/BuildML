"""Configuration and assumption types for Session-facing causal ML.

Causal estimation requires an explicit :class:`CausalAssumptions` object.
EDA / association / feature-importance paths never populate these fields and
must not be treated as identification evidence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from buildml.core.errors import ValidationError

CausalEstimand = Literal["ATE"]
CausalIdentification = Literal["backdoor"]
CausalBackend = Literal["native", "dowhy", "econml"]
NativeCausalMethod = Literal["t_learner", "ipw", "aipw"]
DoWhyCausalMethod = Literal[
    "backdoor_linear",
    "backdoor_propensity_score",
    "backdoor_propensity_weighting",
]
EconMLCausalMethod = Literal["dml", "causal_forest", "policy_tree"]
CausalMethod = Literal[
    "t_learner",
    "ipw",
    "aipw",
    "backdoor_linear",
    "backdoor_propensity_score",
    "backdoor_propensity_weighting",
    "dml",
    "causal_forest",
    "policy_tree",
]
CausalOutcomeKind = Literal["continuous", "binary"]
CausalRefuteKind = Literal[
    "placebo_treatment",
    "random_confounder",
    "random_common_cause",
    "add_unobserved_common_cause",
    "data_subset",
    "placebo_outcome",
]


@dataclass(slots=True)
class CausalAssumptions:
    """Declared identification assumptions for a causal estimand.

    Estimation APIs refuse to run until treatment, outcome, confounders (or an
    explicit empty-confounder waiver), estimand, and the unconfoundedness /
    positivity acknowledgements are set. Instruments are accepted for forward
    compatibility but IV estimation is not implemented in this surface.
    """

    treatment: str
    outcome: str
    confounders: tuple[str, ...]
    estimand: CausalEstimand = "ATE"
    identification: CausalIdentification = "backdoor"
    instruments: tuple[str, ...] = ()
    acknowledge_unconfoundedness: bool = False
    acknowledge_positivity: bool = False
    allow_empty_confounders: bool = False

    def validate(self) -> None:
        """Raise :class:`ValidationError` when the declaration is incomplete.

        Enforces treatment/outcome presence, backdoor ATE scope, non-empty
        confounders (unless waived), and explicit unconfoundedness/positivity
        acknowledgements before any fit or estimate call proceeds.

        Raises
        ------
        ValidationError
            When any required field is missing or inconsistent.
        """
        if not str(self.treatment).strip():
            raise ValidationError(
                "CausalAssumptions.treatment is required. "
                "Causal estimation refuses to run without a declared treatment."
            )
        if not str(self.outcome).strip():
            raise ValidationError(
                "CausalAssumptions.outcome is required. "
                "Causal estimation refuses to run without a declared outcome."
            )
        if self.treatment == self.outcome:
            raise ValidationError(
                "CausalAssumptions.treatment and outcome must be distinct columns."
            )
        if self.estimand != "ATE":
            raise ValidationError(
                f"Unsupported estimand={self.estimand!r}. "
                "This surface currently identifies ATE under backdoor adjustment."
            )
        if self.identification != "backdoor":
            raise ValidationError(
                f"Unsupported identification={self.identification!r}. "
                "Only backdoor adjustment is implemented (no IV / front-door yet)."
            )
        if self.instruments:
            raise ValidationError(
                "Instruments were declared, but IV estimation is not implemented "
                "on this surface. Remove instruments or wait for an IV path; "
                "do not treat unused instruments as identification."
            )
        conf = tuple(self.confounders)
        if any(not str(c).strip() for c in conf):
            raise ValidationError("Confounder names must be non-empty strings.")
        if self.treatment in conf or self.outcome in conf:
            raise ValidationError(
                "Treatment and outcome must not appear in confounders."
            )
        if len(conf) != len(set(conf)):
            raise ValidationError("Confounders must be unique column names.")
        if not conf and not self.allow_empty_confounders:
            raise ValidationError(
                "CausalAssumptions.confounders is empty. Pass an explicit "
                "confounder list for backdoor adjustment, or set "
                "allow_empty_confounders=True to declare that you assume "
                "unconfoundedness with no covariates (strong assumption)."
            )
        if not self.acknowledge_unconfoundedness:
            raise ValidationError(
                "Causal estimation refused: set "
                "acknowledge_unconfoundedness=True to declare that, "
                "conditional on the listed confounders, treatment assignment "
                "is as-good-as-random (no unmeasured confounding). "
                "EDA associations do not satisfy this requirement."
            )
        if not self.acknowledge_positivity:
            raise ValidationError(
                "Causal estimation refused: set acknowledge_positivity=True "
                "to declare that every confounder stratum has a non-zero "
                "probability of both treatment arms (overlap / positivity)."
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialise declared assumptions for Session history and bundles.

        Emits the treatment/outcome/confounder contract and acknowledgement
        flags so checkpoints and walkthrough panels can replay what the caller
        declared without re-running validation.

        Returns
        -------
        dict[str, Any]
            Treatment, outcome, confounders, estimand, and acknowledgement flags.
        """
        return {
            "treatment": self.treatment,
            "outcome": self.outcome,
            "confounders": list(self.confounders),
            "estimand": self.estimand,
            "identification": self.identification,
            "instruments": list(self.instruments),
            "acknowledge_unconfoundedness": self.acknowledge_unconfoundedness,
            "acknowledge_positivity": self.acknowledge_positivity,
            "allow_empty_confounders": self.allow_empty_confounders,
        }

    @classmethod
    def from_mapping(cls, payload: CausalAssumptions | Mapping[str, Any]) -> CausalAssumptions:
        """Build assumptions from a mapping (Session / AI tool kwargs).

        Accepts an existing :class:`CausalAssumptions` instance unchanged; otherwise
        requires ``treatment``, ``outcome``, and ``confounders`` keys with the
        acknowledgement booleans supplied explicitly by the caller.

        Parameters
        ----------
        payload:
            Assumption instance or mapping with causal column contract fields.

        Returns
        -------
        CausalAssumptions
            Parsed assumption object (not yet validated: call ``validate()``).

        Raises
        ------
        ValidationError
            When ``payload`` is not a mapping or omits required keys.
        """
        if isinstance(payload, CausalAssumptions):
            return payload
        if not isinstance(payload, Mapping):
            raise ValidationError(
                "assumptions must be a CausalAssumptions instance or a mapping "
                "with treatment, outcome, confounders, and acknowledgements."
            )
        conf = payload.get("confounders")
        if conf is None:
            raise ValidationError(
                "CausalAssumptions incomplete: 'confounders' key is required "
                "(use [] only with allow_empty_confounders=True)."
            )
        instruments = payload.get("instruments") or ()
        return cls(
            treatment=str(payload.get("treatment") or ""),
            outcome=str(payload.get("outcome") or ""),
            confounders=tuple(str(c) for c in conf),
            estimand=payload.get("estimand", "ATE"),  # type: ignore[arg-type]
            identification=payload.get("identification", "backdoor"),  # type: ignore[arg-type]
            instruments=tuple(str(c) for c in instruments),
            acknowledge_unconfoundedness=bool(
                payload.get("acknowledge_unconfoundedness", False)
            ),
            acknowledge_positivity=bool(payload.get("acknowledge_positivity", False)),
            allow_empty_confounders=bool(payload.get("allow_empty_confounders", False)),
        )


@dataclass(slots=True)
class CausalConfig:
    """User-facing causal estimation knobs (serializable summary)."""

    method: CausalMethod = "aipw"
    backend: CausalBackend = "native"
    bootstrap_samples: int = 200
    random_state: int | None = 0
    clip_propensity: tuple[float, float] = (0.01, 0.99)
    outcome_model: str = "ridge"
    propensity_model: str = "logistic_regression"

    def to_dict(self) -> dict[str, Any]:
        """Serialise user-facing causal knobs for plan metadata and bundles.

        Captures method, backend, bootstrap, clipping, and nuisance model
        choices chosen at fit time for reproducibility in causal bundles.

        Returns
        -------
        dict[str, Any]
            Method, backend, bootstrap, clipping, and nuisance model names.
        """
        return {
            "method": self.method,
            "backend": self.backend,
            "bootstrap_samples": self.bootstrap_samples,
            "random_state": self.random_state,
            "clip_propensity": list(self.clip_propensity),
            "outcome_model": self.outcome_model,
            "propensity_model": self.propensity_model,
        }


def coerce_confounders(confounders: Sequence[str] | None) -> tuple[str, ...] | None:
    """Normalize confounders; ``None`` means 'not declared' (incomplete).

    Converts caller-supplied sequences to a tuple of strings for
    :class:`CausalAssumptions` construction; ``None`` signals the confounder
    list was omitted rather than intentionally empty.

    Parameters
    ----------
    confounders:
        Confounder column names, or ``None`` when not yet declared.

    Returns
    -------
    tuple[str, ...] or None
        Normalised confounder names, or ``None`` when input was ``None``.
    """
    if confounders is None:
        return None
    return tuple(str(c) for c in confounders)
