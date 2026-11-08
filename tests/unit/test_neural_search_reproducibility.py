"""A public search seed controls model initialization, not just fold membership."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import loguniform

from buildml import Session
from buildml.core.errors import MissingExtraError
from buildml.dl.extras import require_torch


@pytest.mark.parametrize("dropout", [0., .3])
@pytest.mark.parametrize("operation", ["search", "cross_validate"])
def test_seeded_neural_evaluation_is_independent_of_ambient_torch_rng(dropout, operation):
    try:
        torch = require_torch(feature="real seeded neural search contract")
    except (MissingExtraError, ImportError, OSError) as exc:
        pytest.skip(str(exc))
    rng = np.random.default_rng(142)
    frame = pd.DataFrame(rng.normal(size=(72, 2)), columns=["x", "z"])
    frame["y"] = (frame.x > frame.z).astype(int)

    def run(ambient_seed):
        session = (Session.ingest(frame).set_roles({"x": "feature", "z": "feature", "y": "target"})
                   .split(test_size=.25, validation_size=.2, random_state=8))
        holdout = session.partition("test").copy()
        starts = []

        def factory(width, task, classes, params=None):
            model = torch.nn.Sequential(torch.nn.Linear(width, 4), torch.nn.ReLU(),
                                        torch.nn.Dropout(dropout), torch.nn.Linear(4, classes))
            starts.append(torch.cat([p.detach().flatten() for p in model.parameters()]).clone())
            return model

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(ambient_seed)
            if operation == "search":
                result = session.dl.search(
                    param_distributions={"learning_rate": loguniform(.001, .01)},
                    n_iter=2, n_folds=2, epochs=1, batch_size=16, device="cpu", seed=71,
                    module_factory=factory,
                )
                assert "test" in result.held_out_partitions
            else:
                from buildml.dl.cv import cross_validate_torch

                # The low-level CV API evaluates every supplied row. Give it
                # only Session train rows, keeping the outer holdout reserved.
                training_data = Session.ingest(session.partition("train")).set_roles(session.dataset.roles).dataset
                result = cross_validate_torch(training_data, n_folds=2, epochs=1,
                                              batch_size=16, device="cpu", seed=71,
                                              module_factory=factory)
        pd.testing.assert_frame_equal(session.partition("test"), holdout)
        return starts, result

    first_states, first = run(9)
    second_states, second = run(917)
    assert len(first_states) == len(second_states) == (4 if operation == "search" else 2)
    for left, right in zip(first_states, second_states, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    if operation == "search":
        assert first.best_params == second.best_params
        assert [trial.params for trial in first.trials] == [trial.params for trial in second.trials]
        for left, right in zip(first.trials, second.trials, strict=True):
            assert left.mean_metrics == pytest.approx(right.mean_metrics)
    else:
        assert first.mean_metrics == pytest.approx(second.mean_metrics)
