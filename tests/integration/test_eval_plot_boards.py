from pathlib import Path

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression

from buildml import Session
from buildml.core.errors import MissingExtraError, ValidationError
from buildml.model.plot_boards import PlotBoardReport


def _has_viz() -> bool:
    try:
        import matplotlib  # noqa: F401
        import seaborn  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(not _has_viz(), reason="viz extra not installed")
def test_eval_plot_board_classification_exports(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "x1": list(range(48)),
            "x2": [i * 0.3 for i in range(48)],
            "y": [0] * 24 + [1] * 24,
        }
    )
    fig_dir = tmp_path / "figs"
    html_path = tmp_path / "board.html"
    session = (
        Session.ingest(frame)
        .set_roles({"x1": "feature", "x2": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
        .scale(method="standard")
        .fit(LogisticRegression(max_iter=500), task="classification")
    )
    board = session.eval_plots(
        export_figures=fig_dir,
        export_html=html_path,
        include_learning_curve=True,
        include_importance=True,
        n_importance_repeats=3,
        learning_curve_cv=3,
    )
    assert isinstance(board, PlotBoardReport)
    assert board.task == "classification"
    assert "confusion_matrix" in board.figure_paths
    assert "roc_curve" in board.figure_paths
    assert "pr_curve" in board.figure_paths
    assert "calibration" in board.figure_paths
    assert "threshold_tradeoff" in board.figure_paths
    assert "learning_curve" in board.figure_paths
    assert "permutation_importance" in board.figure_paths
    assert html_path.exists()
    html = html_path.read_text(encoding="utf-8")
    assert "BuildML Evaluation Plot Board" in html
    assert "Evidence-linked recommendations" in html
    assert "Figure board" in html
    assert board.interpretation


@pytest.mark.skipif(not _has_viz(), reason="viz extra not installed")
def test_eval_plot_board_regression_and_evaluate_hook(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "x1": list(range(40)),
            "x2": [i**0.5 for i in range(40)],
            "y": [2 * i + 1 for i in range(40)],
        }
    )
    fig_dir = tmp_path / "reg_figs"
    session = (
        Session.ingest(frame)
        .set_roles({"x1": "feature", "x2": "feature", "y": "target"})
        .split(test_size=0.25, random_state=0)
        .fit(LinearRegression(), task="regression")
    )
    result = session.evaluate(
        include_plots=True,
        export_figures=fig_dir,
        export_html=tmp_path / "reg.html",
    )
    assert "plot_board" in result.diagnostics
    board = session.last_plot_board
    assert board is not None
    assert "residuals_scatter" in board.figure_paths
    assert "predicted_vs_actual" in board.figure_paths
    # Classification-only panels should be skipped, not crash.
    skipped_panels = {item["panel"] for item in board.skipped}
    assert "roc_curve" in skipped_panels


@pytest.mark.skipif(not _has_viz(), reason="viz extra not installed")
def test_eval_plots_degrade_without_predict_proba(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "x1": list(range(40)),
            "x2": [i * 0.2 for i in range(40)],
            "y": [0] * 20 + [1] * 20,
        }
    )
    # SVC without probability=True lacks predict_proba by default — use RF is fine;
    # instead wrap a classifier that truly lacks proba: sklearn's Perceptron.
    from sklearn.linear_model import Perceptron

    session = (
        Session.ingest(frame)
        .set_roles({"x1": "feature", "x2": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
        .scale()
        .fit(Perceptron(max_iter=500), task="classification")
    )
    board = session.eval_plots(
        export_figures=tmp_path / "no_proba",
        include_learning_curve=False,
        include_importance=False,
    )
    skipped = {item["panel"]: item["reason"] for item in board.skipped}
    assert "roc_curve" in skipped
    assert "predict_proba" in skipped["roc_curve"]
    assert "confusion_matrix" in board.figures


def test_eval_plots_requires_fit() -> None:
    session = Session.ingest(pd.DataFrame({"x": [1, 2], "y": [0, 1]})).set_roles(
        {"x": "feature", "y": "target"}
    )
    with pytest.raises(ValidationError, match="fitted"):
        session.eval_plots()


def test_eval_plots_missing_viz_extra() -> None:
    if _has_viz():
        pytest.skip("viz installed in this environment")
    frame = pd.DataFrame(
        {
            "x1": list(range(20)),
            "y": [0] * 10 + [1] * 10,
        }
    )
    session = (
        Session.ingest(frame)
        .set_roles({"x1": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
        .fit(LogisticRegression(max_iter=300), task="classification")
    )
    with pytest.raises(MissingExtraError):
        session.eval_plots()


@pytest.mark.skipif(not _has_viz(), reason="viz extra not installed")
def test_calibration_html_export_has_findings(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "x1": list(range(40)),
            "x2": [i * 0.5 for i in range(40)],
            "y": [0] * 20 + [1] * 20,
        }
    )
    session = (
        Session.ingest(frame)
        .set_roles({"x1": "feature", "x2": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
        .scale()
        .fit(LogisticRegression(max_iter=500), task="classification")
    )
    html_path = tmp_path / "cal.html"
    cal = session.calibration(export_html=html_path, export_figures=tmp_path / "cal_figs")
    assert "ece" in cal.payload
    assert cal.interpretation
    text = html_path.read_text(encoding="utf-8")
    assert "Top findings" in text or "Brier" in text
    assert "Interpretation" in text


def test_rf_still_builds_proba_board(tmp_path: Path) -> None:
    if not _has_viz():
        pytest.skip("viz extra not installed")
    frame = pd.DataFrame(
        {
            "x1": list(range(36)),
            "x2": [(i % 5) for i in range(36)],
            "y": [0] * 18 + [1] * 18,
        }
    )
    session = (
        Session.ingest(frame)
        .set_roles({"x1": "feature", "x2": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=1)
        .fit(RandomForestClassifier(n_estimators=20, random_state=0), task="classification")
    )
    board = session.eval_plots(
        export_figures=tmp_path / "rf",
        include_learning_curve=False,
        include_importance=False,
    )
    assert "threshold_tradeoff" in board.figure_paths
