"""Exercise a wheel installation outside its source checkout; write JSON evidence."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import platform
import sys
import traceback
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_root.resolve()
    environment_config = Path(sys.prefix) / "pyvenv.cfg"
    evidence = {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "working_directory": str(Path.cwd()),
        "wheel": args.wheel.name,
        "wheel_sha256": hashlib.sha256(args.wheel.read_bytes()).hexdigest(),
        "checks": [],
        "optional_backends": "Not exercised: only core workflows are checked",
        "sdist_installation": "Not exercised by this wheel acceptance runner",
        "venv_configuration": environment_config.read_text(encoding="utf-8")
        if environment_config.exists()
        else None,
        "packages": sorted(f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions()),
    }

    def check(name, operation):
        try:
            operation()
        except Exception:
            evidence["checks"].append(
                {"name": name, "status": "failed", "traceback": traceback.format_exc()}
            )
        else:
            evidence["checks"].append({"name": name, "status": "passed"})

    def provenance():
        import buildml

        assert not Path.cwd().resolve().is_relative_to(source), "Run outside the checkout"
        assert not Path(buildml.__file__).resolve().is_relative_to(source), (
            "Source import escaped installation"
        )
        dist = metadata.distribution("buildml")
        direct = json.loads(dist.read_text("direct_url.json") or "{}")
        assert not direct.get("dir_info", {}).get("editable"), "Editable installation forbidden"
        installed_hash = direct.get("archive_info", {}).get("hashes", {}).get("sha256")
        assert installed_hash == evidence["wheel_sha256"], "Installed artifact does not match wheel"
        evidence["buildml_import"] = str(Path(buildml.__file__).resolve())
        evidence["buildml_version"] = metadata.version("buildml")
        for name, module in tuple(sys.modules.items()):
            if name == "buildml" or name.startswith("buildml."):
                origin = getattr(module, "__file__", None)
                if origin:
                    assert not Path(origin).resolve().is_relative_to(source), name

    check("installed_artifact_provenance", provenance)
    if evidence["checks"][0]["status"] == "passed":
        import numpy as np
        import pandas as pd
        from sklearn.datasets import load_breast_cancer
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier

        from buildml import Session
        from buildml.preprocess import PreprocessRecipe

        def classical():
            frame = pd.DataFrame(
                {
                    "age": [21, None, 35, 40, 29, 33, 52, 47],
                    "income": [40, 55, 60, 80, 50, 70, 90, 65],
                    "approved": [0, 1, 0, 1, 0, 1, 1, 0],
                }
            )
            session = Session.ingest(frame).set_roles(
                {"age": "feature", "income": "feature", "approved": "target"}
            )
            session.split(test_size=0.25, stratify=True, random_state=42)
            session.impute(strategy="median").scale(method="standard")
            session.fit(LogisticRegression(max_iter=500), task="classification")
            assert session.evaluate(partition="test").metrics
            session.explain("split")
            session.learn("leakage")
            session.workflow()
            session.checkpoint_save("checkpoint")
            restored = Session.checkpoint_load("checkpoint", trusted=True)
            pd.testing.assert_frame_equal(restored.to_pandas(), session.to_pandas())
            session.save_pipeline("pipeline", evaluate_partition="test")

        def cross_validation():
            frame = load_breast_cancer(as_frame=True).frame
            session = Session.ingest(frame).set_roles(
                {**{c: "feature" for c in frame if c != "target"}, "target": "target"}
            )
            session.split(test_size=0.2, stratify=True, random_state=42)
            recipe = PreprocessRecipe(impute="median", scale="standard")
            session.cv_score(LogisticRegression(max_iter=500), cv=5, preprocess=recipe)
            session.grid_search(
                DecisionTreeClassifier(random_state=0),
                param_grid={"max_depth": [2, 4, 6], "min_samples_leaf": [1, 5]},
                cv=5,
                preprocess=recipe,
            )

        def eda():
            frame = pd.DataFrame(
                {
                    "x": np.arange(40, dtype=float),
                    "related": np.arange(40) * 2.0,
                    "y": np.arange(40) + 0.3,
                }
            )
            session = Session.ingest(frame).set_roles(
                {"x": "feature", "related": "feature", "y": "target"}
            )
            session.split(test_size=0.25, random_state=1)
            report = session.eda(partition="train", max_plots=0)
            assert report.overview["n_rows"] == 30
            assert report.overview["analysis_partition"] == "train"
            assert set(report.overview["eligible_feature_columns"]) == {"x", "related"}
            assert report.target["summary"]["type"] == "regression_target"
            assert np.isclose(report.bivariate["top_abs_pearson_pairs"][0]["corr"], 1)
            assert report.drift["test_rows"] == 10
            missing = Session.ingest(
                pd.DataFrame({"x": range(10), "y": [0] * 4 + [1] * 4 + [None] * 2})
            ).set_roles({"y": "target"})
            summary = missing.eda(max_plots=0).target["summary"]
            assert summary["n_classes"] == 2 and summary["missing_target_rows"] == 2
            bad = Session.ingest(
                pd.DataFrame({"x": [1.0, np.inf, -np.inf, np.nan], "y": [1.0, 2.0, 3.0, 4.0]})
            ).set_roles({"y": "target"})
            assert bad.eda(max_plots=0).quality["nonfinite_cell_count"] == 2
            no_features = Session.ingest(
                pd.DataFrame({"id": range(40), "y": [0, 1] * 20})
            ).set_roles({"id": "id", "y": "target"})
            no_features.split(test_size=0.25)
            assert no_features.eda(max_plots=0).drift["available"] is False

        def conformal_boundary():
            from buildml.core.errors import ValidationError
            from buildml.probabilistic.conformal import conformal_quantile

            assert conformal_quantile(np.arange(5.0), 1 / 6) == 4.0
            try:
                conformal_quantile(np.arange(5.0), 0.01)
            except ValidationError:
                pass
            else:
                raise AssertionError("Unsupported finite conformal cutoff was accepted")

        def probabilistic_reporting():
            from buildml.core.errors import ValidationError

            rng = np.random.default_rng(9)
            x = rng.normal(size=180)
            y = 2 * x + rng.normal(size=180)
            for task in ("regression", "classification"):
                target = y if task == "regression" else (y > 0).astype(int)
                session = Session.ingest(pd.DataFrame({"x": x, "y": target}))
                session.set_roles({"x": "feature", "y": "target"})
                session.split(test_size=.25, validation_size=.2, random_state=0)
                session.probabilistic.fit(
                    estimator="bayesian_ridge" if task == "regression" else "gaussian_nb",
                    conformal=True, alpha=.1,
                )
                for operation in (session.probabilistic.predict_interval,
                                  session.probabilistic.evaluate):
                    try:
                        operation(alpha=.01)
                    except ValidationError:
                        pass
                    else:
                        raise AssertionError("Stored cutoff accepted a different alpha")
                result = session.probabilistic.evaluate(partition="all")
                assert result.alpha == .1
                assert any("diagnostic" in warning for warning in result.warnings)
                assert any("Evaluation population: all" in text for text in result.disclosures)
                assert any("does not fit or recalibrate" in text for text in result.disclosures)

        check("readme_classical_teaching_checkpoint_pipeline", classical)
        check("readme_fold_local_cv_and_grid_search", cross_validation)
        check("eda_scope_continuous_features_missing_nonfinite_unavailable", eda)
        check("finite_sample_conformal_boundary", conformal_boundary)
        check("probabilistic_alpha_and_population_reporting", probabilistic_reporting)
        check("loaded_module_provenance", provenance)
    evidence["status"] = (
        "passed" if all(c["status"] == "passed" for c in evidence["checks"]) else "failed"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(json.dumps(evidence, indent=2))
    return 0 if evidence["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
