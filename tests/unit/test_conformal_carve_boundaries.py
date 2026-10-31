"""Calibration carving must preserve unique observation identities."""
import numpy as np
import pytest

from buildml.core.errors import ValidationError
from buildml.probabilistic.features import split_train_for_conformal


@pytest.mark.parametrize("indices", [list(range(30)), [f"row-{i}" for i in range(30)],
                                     [("site", i) for i in range(30)],
                                     list(range(15)) + [f"row-{i}" for i in range(15)]])
@pytest.mark.parametrize("seed", [0, 491])
def test_stratified_carve_preserves_disjoint_original_identities_and_class_membership(indices, seed):
    labels = ["common"] * 26 + ["rare"] * 4
    options = dict(calibration_fraction=.2, random_state=seed, stratify_labels=labels)
    fit, calibration = split_train_for_conformal(indices, **options)
    assert len(fit) == 24 and len(calibration) == 6
    assert set(fit).isdisjoint(calibration)
    assert set(fit) | set(calibration) == set(indices)
    label_map = dict(zip(indices, labels, strict=True))
    assert {label_map[i] for i in fit} == {"common", "rare"}
    assert {label_map[i] for i in calibration} == {"common", "rare"}
    assert fit == [i for i in indices if i in set(fit)]
    assert calibration == [i for i in indices if i in set(calibration)]
    assert (fit, calibration) == split_train_for_conformal(indices, **options)


@pytest.mark.parametrize("stratify", [False, True])
def test_repeated_training_observation_is_rejected_before_carving(stratify):
    indices = [0] * 12 + list(range(1, 19))
    with pytest.raises(ValidationError, match="unique"):
        split_train_for_conformal(indices, calibration_fraction=.2, random_state=0,
                                  stratify_labels=[0, 1] * 15 if stratify else None)


@pytest.mark.parametrize("minimum", [0, -1, 1.5, True, None])
@pytest.mark.parametrize("field", ["min_fit", "min_calib"])
def test_invalid_carve_minimums_fail_with_validation_error(field, minimum):
    with pytest.raises(ValidationError, match=field):
        split_train_for_conformal(range(30), calibration_fraction=.2, random_state=0, **{field: minimum})


@pytest.mark.parametrize("fraction", [None, "invalid", 0, .01, .9, np.nan, np.inf])
def test_invalid_calibration_fraction_fails_explicitly(fraction):
    with pytest.raises(ValidationError, match="fraction"):
        split_train_for_conformal(range(30), calibration_fraction=fraction, random_state=0)


def test_rare_singleton_class_and_label_length_mismatch_cannot_claim_a_valid_carve():
    with pytest.raises(ValidationError, match="at least 2"):
        split_train_for_conformal(range(30), calibration_fraction=.2, random_state=0,
                                  stratify_labels=["common"] * 29 + ["singleton"])
    with pytest.raises(ValidationError, match="length"):
        split_train_for_conformal(range(30), calibration_fraction=.2, random_state=0,
                                  stratify_labels=["common"] * 20)
