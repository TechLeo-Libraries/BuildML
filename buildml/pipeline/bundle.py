"""Save the model *and* the transforms it was trained on, as one artifact.

Saving an estimator by itself is the most common way a model that worked in
training fails in production. The estimator learned from data that had been
imputed with training medians, encoded with training categories, and scaled by
training statistics. Hand it a raw row and it will happily return a number that
means nothing, because nothing checked that the row was prepared the same way.

A pipeline bundle is the estimator plus every fitted plan needed to prepare a
row, plus a schema contract describing what the input must look like, plus a
model card recording where it came from. Restoring it restores the whole path
from a raw record to a prediction.

The layout is a directory: ``model.joblib``, ``plans.joblib``, ``meta.json``,
``schema_contract.json``, and the model card in both JSON and Markdown. The card
in Markdown is there so a human can read it without tooling.

Both files carry format versions, and both loaders accept their predecessors, so
a bundle written by an older BuildML keeps working.

See Also
--------
buildml.pipeline.score : Turning a loaded bundle into predictions.
buildml.pipeline.contract : What the input must look like, and enforcing it.
buildml.checkpoint : The complementary artifact, for resuming rather than serving.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

from buildml._version import __version__
from buildml.core.errors import ValidationError
from buildml.core.serialization import (
    assert_local_load_path,
    joblib_load_trusted,
    read_json_sidecar,
    sha256_file,
)
from buildml.model.supervised import FitResult
from buildml.pipeline.card import ModelCard, build_model_card, load_model_card, save_model_card
from buildml.pipeline.contract import (
    SCHEMA_CONTRACT_FILENAME,
    SchemaContract,
    build_schema_contract,
    input_columns_from_plans,
    load_schema_contract,
    save_schema_contract,
)
from buildml.preprocess.binning import BinningPlan
from buildml.preprocess.custom import CustomTransformPlan
from buildml.preprocess.dates import DateFeaturePlan
from buildml.preprocess.encode import EncodePlan
from buildml.preprocess.imbalance import ResamplePlan
from buildml.preprocess.impute import SimpleImputePlan
from buildml.preprocess.outliers import OutlierPlan
from buildml.preprocess.reduce import ReducePlan
from buildml.preprocess.scale import ScalePlan
from buildml.preprocess.select import FeatureSelectPlan
from buildml.preprocess.text import TextFeaturePlan

# Bundle directory meta.json format. v1 exploratory bundles remain readable.
BUNDLE_FORMAT = "buildml.pipeline_bundle.v2"
BUNDLE_FORMAT_V1 = "buildml.pipeline_bundle.v1"
SUPPORTED_BUNDLE_FORMATS = {BUNDLE_FORMAT, BUNDLE_FORMAT_V1, None}
# plans.joblib envelope. Unversioned dicts with plan keys are treated as v1.
PLANS_FORMAT = "buildml.plans.v2"
PLANS_FORMAT_V1 = "buildml.plans.v1"
CHECKPOINT_COMPATIBILITY = (
    "Pipeline bundles and checkpoints are complementary, not interchangeable. "
    "A pipeline bundle stores fitted preprocess plans, the estimator, and a model card; "
    "it does not embed dataset rows, split indices, or full Session history. "
    "A checkpoint stores data, roles, splits, history, and optional plan metadata; "
    "it does not embed the estimator. Store them side by side when both resume and "
    "inference are required. Reload plans+model via load_pipeline_bundle; reload data "
    "via checkpoint_load. Bundle meta format is buildml.pipeline_bundle.v2; "
    "plans.joblib uses buildml.plans.v2 with a migration path for older unversioned "
    "plan dicts."
)


@dataclass(slots=True)
class PipelineBundle:
    """Everything needed to turn a raw row into a prediction, in one object.

    The plan fields are each ``None`` when that step was not part of training.
    Their *order* matters and is not encoded here — it is fixed by
    :func:`~buildml.pipeline.score.predict_from_pipeline`, which must apply them
    exactly as they were applied during training. Scaling before encoding
    produces different numbers than encoding before scaling.

    Attributes
    ----------
    fit_result:
        The estimator, its task, and the feature columns it expects in order.
    impute_plan, encode_plan, scale_plan, date_plan:
        The fitted transforms, each holding the values learned from training
        data — the medians, the category maps, the means and scales. These are
        what make a prediction reproducible.
    outlier_plan, binning_plan, feature_select_plan:
        As above — outlier handling, binning, and feature selection.
    text_plan, reduce_plan, custom_plan:
        As above — text featurisation, dimensionality reduction, and any
        user-supplied transform.
    resample_plan:
        Kept for lineage only. Resampling rewrites training rows to rebalance
        classes; there is nothing to apply at inference, and applying it would
        be wrong. Recorded so the model card can say the model was trained on
        resampled data, which changes how its probabilities should be read.
    model_card:
        Provenance — data, metrics, history, and limitations.
    schema_contract:
        What an input frame must contain, checked at score time.
    plans_format, bundle_format:
        Version labels, so a future loader knows what it is reading.

    Notes
    -----
    **A bundle is not a checkpoint.** It holds no data rows, no split indices,
    and no full session history, so it cannot resume an analysis. Save both when
    a run needs to be both resumable and deployable.

    **The plans hold statistics computed from training data**, which is what
    makes them safe. Refitting them on incoming data at score time would
    reintroduce exactly the drift the bundle exists to prevent.

    See Also
    --------
    save_pipeline_bundle : Writing one.
    load_pipeline_bundle : Reading one back.
    """

    fit_result: FitResult
    impute_plan: SimpleImputePlan | None = None
    encode_plan: EncodePlan | None = None
    scale_plan: ScalePlan | None = None
    date_plan: DateFeaturePlan | None = None
    outlier_plan: OutlierPlan | None = None
    binning_plan: BinningPlan | None = None
    feature_select_plan: FeatureSelectPlan | None = None
    text_plan: TextFeaturePlan | None = None
    reduce_plan: ReducePlan | None = None
    custom_plan: CustomTransformPlan | None = None
    resample_plan: ResamplePlan | None = None
    model_card: ModelCard | None = None
    schema_contract: SchemaContract | None = None
    plans_format: str = PLANS_FORMAT
    bundle_format: str = BUNDLE_FORMAT

    def to_meta(self) -> dict[str, Any]:
        """Describe the bundle as JSON, for ``meta.json``.

        This is the human-readable index of the bundle. The plans appear here as
        dictionaries — their configuration and learned values — while the
        objects that actually do the work live in ``plans.joblib``. Anyone can
        read ``meta.json`` to see what a bundle contains and what it was trained
        with, without unpickling anything.

        Returns
        -------
        dict
            The format versions, the BuildML version that wrote it, the fit
            summary, each plan as a dictionary or ``None``, flags for the card
            and contract, and the compatibility note explaining how bundles
            relate to checkpoints.

        Notes
        -----
        **The model card and contract are flagged, not embedded.** They have
        their own files, and duplicating them here would create two copies that
        could disagree.
        """
        return {
            "format": self.bundle_format,
            "plans_format": self.plans_format,
            "buildml_version": __version__,
            "fit": self.fit_result.to_dict(),
            "impute_plan": None if self.impute_plan is None else self.impute_plan.to_dict(),
            "encode_plan": None if self.encode_plan is None else self.encode_plan.to_dict(),
            "scale_plan": None if self.scale_plan is None else self.scale_plan.to_dict(),
            "date_plan": None if self.date_plan is None else self.date_plan.to_dict(),
            "outlier_plan": None if self.outlier_plan is None else self.outlier_plan.to_dict(),
            "binning_plan": None if self.binning_plan is None else self.binning_plan.to_dict(),
            "feature_select_plan": (
                None if self.feature_select_plan is None else self.feature_select_plan.to_dict()
            ),
            "text_plan": None if self.text_plan is None else self.text_plan.to_dict(),
            "reduce_plan": None if self.reduce_plan is None else self.reduce_plan.to_dict(),
            "custom_plan": None if self.custom_plan is None else self.custom_plan.to_dict(),
            "resample_plan": None if self.resample_plan is None else self.resample_plan.to_dict(),
            "has_model_card": self.model_card is not None,
            "has_schema_contract": self.schema_contract is not None,
            "schema_contract_file": SCHEMA_CONTRACT_FILENAME,
            "compatibility": CHECKPOINT_COMPATIBILITY,
        }


def pack_plans_payload(
    *,
    impute_plan: SimpleImputePlan | None = None,
    encode_plan: EncodePlan | None = None,
    scale_plan: ScalePlan | None = None,
    date_plan: DateFeaturePlan | None = None,
    outlier_plan: OutlierPlan | None = None,
    binning_plan: BinningPlan | None = None,
    feature_select_plan: FeatureSelectPlan | None = None,
    text_plan: TextFeaturePlan | None = None,
    reduce_plan: ReducePlan | None = None,
    custom_plan: CustomTransformPlan | None = None,
    resample_plan: ResamplePlan | None = None,
) -> dict[str, Any]:
    """Wrap the plan objects in a versioned envelope before pickling them.

    The envelope is the reason old bundles keep loading. A bare dictionary of
    plans has no way to say which layout it uses, so adding or renaming a plan
    kind later becomes a guessing game for the loader. Recording a format label
    and the writing version alongside the plans makes migration mechanical.

    Every plan key is present in the payload even when its value is ``None``, so
    a reader sees the full set of possible steps rather than inferring absence
    from a missing key.

    Parameters
    ----------
    impute_plan, encode_plan, scale_plan, date_plan:
        The fitted transforms to store, each ``None`` when that step was not
        used.
    outlier_plan, binning_plan, feature_select_plan:
        As above — outlier handling, binning, and feature selection.
    text_plan, reduce_plan, custom_plan:
        As above — text featurisation, dimensionality reduction, and any
        user-supplied transform.
    resample_plan:
        Stored for lineage only; never applied at inference.

    Returns
    -------
    dict
        ``{'format', 'buildml_version', 'plans': {...}}``, ready for
        :func:`joblib.dump`.

    Notes
    -----
    **The plans are stored as live objects, not serialised dictionaries**, which
    is why this goes through joblib rather than JSON — the transforms have to be
    callable again after loading.

    See Also
    --------
    unpack_plans_payload : The inverse, including the migration path.
    """
    return {
        "format": PLANS_FORMAT,
        "buildml_version": __version__,
        "plans": {
            "impute_plan": impute_plan,
            "encode_plan": encode_plan,
            "scale_plan": scale_plan,
            "date_plan": date_plan,
            "outlier_plan": outlier_plan,
            "binning_plan": binning_plan,
            "feature_select_plan": feature_select_plan,
            "text_plan": text_plan,
            "reduce_plan": reduce_plan,
            "custom_plan": custom_plan,
            "resample_plan": resample_plan,
        },
    }


def unpack_plans_payload(loaded: Any) -> tuple[dict[str, Any], str]:
    """Read either payload layout and return one shape, plus which it was.

    Two layouts exist in the wild. The current one is the versioned envelope
    written by :func:`pack_plans_payload`. The older one is a flat dictionary
    with the plan keys at the top level, written before versioning existed.
    Both are accepted and normalised to the same result, so callers never branch
    on the layout.

    The format label is returned rather than discarded because it belongs in the
    loaded bundle: knowing an artifact came from an older writer is useful when
    a prediction looks wrong.

    Parameters
    ----------
    loaded:
        Whatever :func:`joblib.load` produced from ``plans.joblib``. Typed
        loosely because that is genuinely unknown until inspected.

    Returns
    -------
    tuple of (dict, str)
        The plan dictionary with every key present — missing plans as ``None``
        — and the detected format label.

    Raises
    ------
    ValidationError
        If the payload is not a mapping, carries an unrecognised format label,
        or is a mapping with no plan keys at all. The last case is the one worth
        failing on: a payload that yielded silently empty plans would produce a
        bundle that predicts from raw, unprepared rows.

    Notes
    -----
    **The empty-plan template is applied first, then updated.** This guarantees
    the full key set regardless of which layout was read, so downstream code can
    index without ``get``.

    See Also
    --------
    pack_plans_payload : The writer.
    """
    empty = {
        "impute_plan": None,
        "encode_plan": None,
        "scale_plan": None,
        "date_plan": None,
        "outlier_plan": None,
        "binning_plan": None,
        "feature_select_plan": None,
        "text_plan": None,
        "reduce_plan": None,
        "custom_plan": None,
        "resample_plan": None,
    }
    if not isinstance(loaded, dict):
        raise ValidationError("plans.joblib payload must be a mapping")
    fmt = loaded.get("format")
    if fmt == PLANS_FORMAT or (
        fmt is None and "plans" in loaded and isinstance(loaded["plans"], dict)
    ):
        if fmt not in {PLANS_FORMAT, None}:
            raise ValidationError(f"Unsupported plans.joblib format '{fmt}'")
        plans = dict(empty)
        plans.update({k: loaded["plans"].get(k) for k in empty})
        return plans, PLANS_FORMAT if fmt == PLANS_FORMAT else PLANS_FORMAT_V1
    # Flat v1 / legacy: plan keys at the top level.
    plan_keys = set(empty)
    if plan_keys.intersection(loaded.keys()):
        plans = dict(empty)
        plans.update({k: loaded.get(k) for k in empty})
        return plans, PLANS_FORMAT_V1 if fmt in {None, PLANS_FORMAT_V1} else str(fmt)
    raise ValidationError(
        "Unrecognized plans.joblib payload. Expected a buildml.plans.v2 envelope "
        "or a flat dict with impute_plan/encode_plan/... keys."
    )


def save_pipeline_bundle(
    path: str | Path,
    *,
    fit_result: FitResult,
    impute_plan: SimpleImputePlan | None = None,
    encode_plan: EncodePlan | None = None,
    scale_plan: ScalePlan | None = None,
    date_plan: DateFeaturePlan | None = None,
    outlier_plan: OutlierPlan | None = None,
    binning_plan: BinningPlan | None = None,
    feature_select_plan: FeatureSelectPlan | None = None,
    text_plan: TextFeaturePlan | None = None,
    reduce_plan: ReducePlan | None = None,
    custom_plan: CustomTransformPlan | None = None,
    resample_plan: ResamplePlan | None = None,
    model_card: ModelCard | None = None,
    dataset_schema: dict[str, Any] | None = None,
    roles: dict[str, Any] | None = None,
    input_columns: list[str] | tuple[str, ...] | None = None,
    schema_contract: SchemaContract | None = None,
    history: list[dict[str, Any]] | None = None,
    metrics: dict[str, dict[str, float]] | None = None,
    title: str | None = None,
) -> Path:
    """Write the estimator, its transforms, a contract, and a card to a directory.

    The one call that turns a fitted model into something deployable. Beyond
    writing the files, it derives two things you would otherwise have to supply.

    The schema contract is inferred from the plans: each fitted transform knows
    which columns it consumes, so the union of those is what an input frame must
    provide. When no plans were used, the estimator's own feature columns become
    the contract. Either way, a score-time frame can be checked before it
    reaches the model, and a missing or misnamed column becomes a clear error
    rather than a wrong prediction.

    The model card is built from the fit result, the plan summaries, and
    whatever history and metrics were passed. Providing ``metrics`` and
    ``history`` is worth the effort — six months on, a card that records what
    the model scored and how it got there is the difference between trusting an
    artifact and rebuilding it.

    Parameters
    ----------
    path:
        The destination directory, created if missing. An existing bundle is
        overwritten file by file, so a partial failure can leave a mixed
        directory; write to a fresh path when that matters.
    fit_result:
        The fitted estimator with its task and feature columns. Required.
    impute_plan, encode_plan, scale_plan, date_plan:
        The fitted transforms to save. Omitting one that was used in training
        produces a bundle that silently under-prepares its inputs, which is the
        main way to get a bundle wrong.
    outlier_plan, binning_plan, feature_select_plan:
        As above — outlier handling, binning, and feature selection.
    text_plan, reduce_plan, custom_plan:
        As above — text featurisation, dimensionality reduction, and any
        user-supplied transform.
    resample_plan:
        Recorded for lineage, never replayed at inference.
    model_card:
        A card to use as-is. When omitted, one is built.
    dataset_schema:
        Column dtypes at training time, used for the contract's type
        expectations and recorded on the card.
    roles:
        Column roles at training time, recorded on the contract.
    input_columns:
        Override the inferred input columns. Rarely needed — inference from the
        plans is usually more accurate than a hand-written list, which drifts.
    schema_contract:
        A contract to use as-is instead of building one.
    history:
        The operation log, for the card's provenance section.
    metrics:
        Scores by partition, for example ``{'test': {'roc_auc': 0.84}}``.
        Recorded on the card as the model's stated performance.
    title:
        A human-readable name for the card.

    Returns
    -------
    Path
        The bundle directory.

    Raises
    ------
    ValidationError
        If ``fit_result`` is ``None``. There is no meaningful bundle without an
        estimator.
    OSError
        If the directory cannot be created or written.

    Notes
    -----
    **The contract is only as good as the plans you pass.** Inferred input
    columns come from the plans that are present, so an omitted plan narrows the
    contract as well as skipping a transform.

    **A bundle does not carry data or splits.** For resuming an analysis, save a
    checkpoint alongside it.

    **Old bundles without ``schema_contract.json`` still load**, with score-time
    checks skipped and a warning. New bundles always write one.

    Examples
    --------
    Save a fitted model with everything needed to serve and to audit it::

        save_pipeline_bundle(
            "artifacts/churn-v3",
            fit_result=fit,
            impute_plan=impute,
            encode_plan=encode,
            scale_plan=scale,
            dataset_schema=dataset.schema.to_dict(),
            metrics={"test": evaluation.metrics},
            history=session.history(),
            title="Churn model v3",
        )

    See Also
    --------
    load_pipeline_bundle : Reading it back.
    buildml.pipeline.score.predict_from_pipeline : Using it.
    """
    if fit_result is None:
        raise ValidationError("fit_result is required to save a pipeline bundle")
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    preprocess_summary = {
        "impute": None if impute_plan is None else impute_plan.to_dict(),
        "encode": None if encode_plan is None else encode_plan.to_dict(),
        "scale": None if scale_plan is None else scale_plan.to_dict(),
        "dates": None if date_plan is None else date_plan.to_dict(),
        "outliers": None if outlier_plan is None else outlier_plan.to_dict(),
        "binning": None if binning_plan is None else binning_plan.to_dict(),
        "feature_select": None if feature_select_plan is None else feature_select_plan.to_dict(),
        "text": None if text_plan is None else text_plan.to_dict(),
        "reduce": None if reduce_plan is None else reduce_plan.to_dict(),
        "custom": None if custom_plan is None else custom_plan.to_dict(),
        "resample": None if resample_plan is None else resample_plan.to_dict(),
    }
    plan_inputs = input_columns_from_plans(
        {
            "impute_plan": impute_plan,
            "encode_plan": encode_plan,
            "scale_plan": scale_plan,
            "date_plan": date_plan,
            "outlier_plan": outlier_plan,
            "binning_plan": binning_plan,
            "feature_select_plan": feature_select_plan,
            "text_plan": text_plan,
            "reduce_plan": reduce_plan,
            "custom_plan": custom_plan,
        }
    )
    resolved_inputs = input_columns
    if resolved_inputs is None and plan_inputs:
        resolved_inputs = plan_inputs
    if resolved_inputs is None:
        # No preprocess plans: score frame must already match the estimator contract.
        resolved_inputs = list(fit_result.feature_columns)
    contract = schema_contract or build_schema_contract(
        schema=dataset_schema,
        roles=roles,
        feature_columns=fit_result.feature_columns,
        target_column=fit_result.target_column,
        input_columns=resolved_inputs,
    )
    card = model_card or build_model_card(
        fit_result=fit_result,
        dataset_schema=dataset_schema,
        preprocess_summary=preprocess_summary,
        history=history,
        metrics=metrics,
        title=title,
        lineage={
            "artifact": "pipeline_bundle",
            "format": BUNDLE_FORMAT,
            "plans_format": PLANS_FORMAT,
            "contains_checkpoint": False,
            "contains_raw_dataset": False,
            "has_schema_contract": True,
            "checkpoint_compatibility": CHECKPOINT_COMPATIBILITY,
            "plans_present": sorted(
                key for key, value in preprocess_summary.items() if value is not None
            ),
        },
    )
    model_path = root / "model.joblib"
    plans_path = root / "plans.joblib"
    joblib.dump(fit_result.estimator, model_path)
    joblib.dump(
        pack_plans_payload(
            impute_plan=impute_plan,
            encode_plan=encode_plan,
            scale_plan=scale_plan,
            date_plan=date_plan,
            outlier_plan=outlier_plan,
            binning_plan=binning_plan,
            feature_select_plan=feature_select_plan,
            text_plan=text_plan,
            reduce_plan=reduce_plan,
            custom_plan=custom_plan,
            resample_plan=resample_plan,
        ),
        plans_path,
    )
    save_schema_contract(root, contract)
    bundle = PipelineBundle(
        fit_result=fit_result,
        impute_plan=impute_plan,
        encode_plan=encode_plan,
        scale_plan=scale_plan,
        date_plan=date_plan,
        outlier_plan=outlier_plan,
        binning_plan=binning_plan,
        feature_select_plan=feature_select_plan,
        text_plan=text_plan,
        reduce_plan=reduce_plan,
        custom_plan=custom_plan,
        resample_plan=resample_plan,
        model_card=card,
        schema_contract=contract,
        plans_format=PLANS_FORMAT,
        bundle_format=BUNDLE_FORMAT,
    )
    meta = bundle.to_meta()
    meta["payload_hashes"] = {
        "model.joblib": sha256_file(model_path),
        "plans.joblib": sha256_file(plans_path),
    }
    (root / "meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    save_model_card(root, card)
    return root


def load_pipeline_bundle(path: str | Path, *, trusted: bool = False) -> PipelineBundle:
    """Restore an estimator together with the transforms it depends on.

    Reads the current format and migrates the older ones, so a bundle written by
    a previous BuildML still loads. The only hard requirements are
    ``model.joblib`` and ``meta.json``; a bundle missing either is incomplete
    and rejected rather than partially restored.

    Everything else degrades. Missing plans give a bundle with ``None`` in those
    slots — correct for a model trained without them, and wrong for one whose
    plans were lost, which is why the plan set is worth checking against the
    card. A missing contract disables score-time validation with a warning.

    Parameters
    ----------
    path:
        The bundle directory.
    trusted:
        Must be ``True`` to deserialize pickle/joblib/torch payloads. Pass
        only for artifacts you created or fully trust. Defaults to ``False``.

    Returns
    -------
    PipelineBundle
        The estimator, the fitted plans, the card and contract when present, and
        the detected format labels.

    Raises
    ------
    ValidationError
        If ``model.joblib`` or ``meta.json`` is absent, if the metadata carries
        an unrecognised format with no recoverable fit section, or if
        ``plans.joblib`` holds an unreadable payload.

    Notes
    -----
    **Loading unpickles an estimator, which executes code.** Only load bundles
    from a source you trust — this is a property of joblib and pickle, not of
    BuildML.

    **A version mismatch in scikit-learn may warn or fail.** Estimators are
    pickled objects tied to the library that created them; record the
    environment alongside the artifact when it has to survive an upgrade.

    See Also
    --------
    save_pipeline_bundle : Writing one.
    buildml.pipeline.score.predict_from_pipeline : Predicting from one.
    """
    root = assert_local_load_path(path, artifact="pipeline bundle")
    model_path = root / "model.joblib"
    plans_path = root / "plans.joblib"
    meta_path = root / "meta.json"
    if not model_path.exists() or not meta_path.exists():
        raise ValidationError(f"Pipeline bundle incomplete at '{root}'")
    meta = read_json_sidecar(meta_path, artifact="pipeline meta.json")
    fmt = meta.get("format")
    if fmt not in SUPPORTED_BUNDLE_FORMATS and "fit" not in meta:
        raise ValidationError(f"Unrecognized pipeline bundle format '{fmt}' at '{root}'")
    payload_hashes = meta.get("payload_hashes")
    model_hash = None
    plans_hash = None
    if isinstance(payload_hashes, dict):
        model_hash = payload_hashes.get("model.joblib")
        plans_hash = payload_hashes.get("plans.joblib")
    estimator = joblib_load_trusted(
        model_path,
        trusted=trusted,
        artifact="joblib plan",
        expected_sha256=model_hash if isinstance(model_hash, str) else None,
    )
    fit_meta = meta["fit"]
    fit_result = FitResult(
        estimator=estimator,
        task=fit_meta["task"],
        feature_columns=tuple(fit_meta["feature_columns"]),
        target_column=fit_meta["target_column"],
        n_train_rows=int(fit_meta["n_train_rows"]),
    )
    plans: dict[str, Any] = {
        "impute_plan": None,
        "encode_plan": None,
        "scale_plan": None,
        "date_plan": None,
        "outlier_plan": None,
        "binning_plan": None,
        "feature_select_plan": None,
        "text_plan": None,
        "reduce_plan": None,
        "custom_plan": None,
        "resample_plan": None,
    }
    plans_format = PLANS_FORMAT_V1
    if plans_path.exists():
        loaded = joblib_load_trusted(
            plans_path,
            trusted=trusted,
            artifact="joblib plan",
            expected_sha256=plans_hash if isinstance(plans_hash, str) else None,
        )
        plans, plans_format = unpack_plans_payload(loaded)
    card = None
    card_path = root / "model_card.json"
    if card_path.exists():
        card = load_model_card(root)
    contract = load_schema_contract(root)
    return PipelineBundle(
        fit_result=fit_result,
        impute_plan=plans.get("impute_plan"),
        encode_plan=plans.get("encode_plan"),
        scale_plan=plans.get("scale_plan"),
        date_plan=plans.get("date_plan"),
        outlier_plan=plans.get("outlier_plan"),
        binning_plan=plans.get("binning_plan"),
        feature_select_plan=plans.get("feature_select_plan"),
        text_plan=plans.get("text_plan"),
        reduce_plan=plans.get("reduce_plan"),
        custom_plan=plans.get("custom_plan"),
        resample_plan=plans.get("resample_plan"),
        model_card=card,
        schema_contract=contract,
        plans_format=plans_format,
        bundle_format=fmt or BUNDLE_FORMAT_V1,
    )
