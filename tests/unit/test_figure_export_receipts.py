"""Figure receipts describe PNGs written during the current export."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from buildml import Session
from buildml.eda.visualize import save_figures


@pytest.fixture
def viz():
    matplotlib = pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    previous = set(plt.get_fignums())
    yield plt
    for number in set(plt.get_fignums()) - previous:
        plt.close(number)


def _session(*, media=False):
    x = np.linspace(0, 1, 40)
    frame = pd.DataFrame({"x": x, "z": np.cos(5 * x), "y": 2.5 * x + 0.4})
    if media:
        frame["group"] = ["east", "west"] * 20
        frame["observed_at"] = pd.date_range(
            "2024-03-09", periods=40, freq="h", tz="America/New_York"
        )
    return Session.ingest(frame).set_roles({"y": "target"}).split(test_size=0.25, random_state=0)


@pytest.mark.parametrize(
    ("consumer", "blocked"),
    [("eda", "00_missingness_matrix"), ("board", "residuals_scatter"), ("curve", "learning_curve")],
)
def test_consumer_reports_only_successfully_written_pngs(viz, tmp_path, consumer, blocked):
    root = tmp_path / "figures"
    obstruction = root / f"{blocked}.png"
    obstruction.mkdir(parents=True)
    sentinel = obstruction / "existing.txt"
    sentinel.write_text("prior file", encoding="utf-8")
    session = _session(media=consumer == "eda")
    if consumer == "eda":
        report = session.eda(include_plots=True, export_figures=root)
    elif consumer == "board":
        session.fit(LinearRegression(), task="regression")
        report = session.eval_plots(
            export_figures=root, include_learning_curve=False, include_importance=False
        )
    else:
        report = session.classical.learning_curve(LinearRegression(), cv=3, export_figures=root)

    assert Path(report.figure_dir) == root
    assert blocked not in report.figure_paths
    assert sentinel.read_text(encoding="utf-8") == "prior file"
    written = {path.stem: path for path in root.glob("*.png") if path.is_file()}
    assert {name: Path(path) for name, path in report.figure_paths.items()} == written
    if consumer == "eda":
        assert any(name.endswith("_temporal_density") for name in written)
        assert any(name.endswith("_target_vs_categorical") for name in written)
    elif consumer == "board":
        assert "predicted_vs_actual" in written
    else:
        assert not written
    for path in written.values():
        assert viz.imread(path).size > 0


def test_direct_export_preserves_previous_png_on_render_failure(viz, tmp_path):
    fig, ax = viz.subplots()
    ax.plot([0, 1], [1, 0])
    assert save_figures({"good": fig}, tmp_path) == tmp_path
    path = tmp_path / "good.png"
    original = path.read_bytes()

    # Invalid math text reaches the PNG renderer and fails during drawing.
    ax.set_title(r"$\buildmlUnknownCommand$")
    valid, valid_ax = viz.subplots()
    valid_ax.plot([0, 1], [0, 1])
    assert (
        save_figures({"good": fig, "new": valid, "error": {"error": "render failed"}}, tmp_path)
        == tmp_path
    )
    assert path.read_bytes() == original
    assert viz.imread(tmp_path / "new.png").size > 0
    assert sorted(p.name for p in tmp_path.iterdir()) == ["good.png", "new.png"]
