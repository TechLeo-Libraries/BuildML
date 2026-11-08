"""Unknown or missing logged actions cannot replace a successful evaluation."""

import joblib
import pandas as pd
import pytest
from test_rl_action_identity_consumers import METHODS, _fit, _frame, _session

from buildml.core.errors import ValidationError


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("bad_action", ["unseen", None])
def test_invalid_selected_actions_preserve_previous_policy_reports_and_history(method, bad_action):
    session = _session(_frame(["left", "right"]))
    _fit(session, method)
    imitation = method in METHODS[:2]
    if imitation:
        session.rl.predict_imitation(partition="test")
        session.rl.evaluate_imitation(partition="test")
    else:
        session.rl.act(partition="test")
        session.rl.evaluate(partition="test")

    def fingerprint():
        return joblib.hash((session.imitation_plan, session.rl_plan, session.history,
                            session._imitation_eval_result, session._imitation_predict_result,
                            session._rl_act_result, session._rl_eval_result))

    before = fingerprint()
    data = session.dataset._ensure_pandas()
    values = list(data.action)
    values[session.split_plan.indices_for("test")[0]] = bad_action
    data["action"] = pd.array(values, dtype=object)
    with pytest.raises(ValidationError, match="action"):
        if imitation:
            session.rl.evaluate_imitation(partition="test")
        else:
            session.rl.evaluate(partition="test")
    assert fingerprint() == before
