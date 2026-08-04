# ruff: noqa: E501
"""Meta-learning (tabular few-shot / episodic) concept notes."""

from __future__ import annotations

from buildml.explain.concepts._builder import _note
from buildml.explain.schemas import ConceptNote

METALEARNING_NOTES: dict[str, ConceptNote] = {
    note.key: note
    for note in (
        _note(
            key="metalearning-episodic",
            title="Episodic few-shot meta-learning on Session tasks",
            summary="Carve tasks via a task/group column; meta-train on train; evaluate with support/query episodes: not foundation-model MAML-at-scale.",
            definition=(
                "Meta-learning in BuildML treats rows sharing a task/group id as "
                "one episodic task. Meta-train runs few-shot support/query "
                "episodes on the train partition only. Evaluation prefers novel "
                "task ids on holdout partitions."
            ),
            intuition=(
                "Each client or dataset slice is a mini exam: study a few labeled "
                "examples (support), then answer the rest (query)."
            ),
            formal_idea=(
                "Tasks τ ~ p(τ); episode (S_τ, Q_τ); adapt θ' = A(θ, S_τ); "
                "score on Q_τ. Train partition only for meta-train."
            ),
            why_it_matters=(
                "Pooling all rows without task structure hides few-shot failure modes.",
                "Using holdout rows during meta-train is leakage.",
            ),
            how_buildml_uses=(
                "session.metalearning.fit → session.metalearning.adapt / session.metalearning.evaluate.",
                "Task column from role='group' or task_column=.",
            ),
            interpretation_rules=(
                "Read n_meta_train_tasks, meta_train_accuracy, novel_task_ids, disclosures.",
                "Overlapping task ids on holdout are disclosed as not true out-of-task tests.",
            ),
            assumptions=(
                "Exactly one target; a task/group column; numeric non-null features.",
            ),
            failure_modes=(
                "Fewer than two train tasks; insufficient per-class rows for k-shot.",
            ),
            anti_patterns=(
                "Claiming foundation-model or MAML-at-scale results from tabular prototypes.",
            ),
            worked_example_pattern=(
                "session.metalearning.fit(method='prototypical', k_shot=5) → "
                "session.metalearning.evaluate('validation').",
            ),
            related_concepts=(
                "metalearning-prototypical",
                "metalearning-warm-start",
                "metalearning-bundle-boundary",
                "leakage-boundary",
            ),
        ),
        _note(
            key="metalearning-prototypical",
            title="Tabular prototypical (nearest-centroid) few-shot",
            summary="Class prototypes = mean support features; query classified by nearest prototype: no learned neural embedding.",
            definition=(
                "method='prototypical' builds a mean feature vector per class from "
                "the support set and assigns query rows by euclidean nearest "
                "centroid. Features may already be scaled/reduced by the Session."
            ),
            intuition=(
                "Sketch the average look of each class from a few examples, then "
                "match new rows to the closest sketch."
            ),
            formal_idea=(
                "c_k = mean({x ∈ S : y=k}); ŷ = argmin_k ||x − c_k||₂."
            ),
            why_it_matters=(
                "A complete, honest few-shot baseline without pretending a neural ProtoNet.",
            ),
            how_buildml_uses=(
                "session.metalearning.fit(method='prototypical'); session.metalearning.adapt builds prototypes.",
            ),
            interpretation_rules=(
                "Read meta_train_accuracy and per-task episodic accuracy/F1.",
            ),
            assumptions=("Classification labels; enough rows per class for k_shot + query.",),
            failure_modes=("Tasks with a single class or fewer than k_shot+1 rows per class.",),
            anti_patterns=("Marketing this as a learned ProtoNet embedding.",),
            worked_example_pattern=(
                "session.metalearning.fit(method='prototypical', k_shot=3, n_episodes=30).",
            ),
            related_concepts=("metalearning-episodic", "metalearning-warm-start"),
        ),
        _note(
            key="metalearning-warm-start",
            title="Warm-start meta-initialization + support adapt",
            summary="Pooled sklearn classifier as meta-init; session.metalearning.adapt clones and refits on the support set.",
            definition=(
                "method='warm_start' fits a logistic / SGD classifier on pooled "
                "meta-train rows, then fast-adapts by cloning and refitting on "
                "each task's support set."
            ),
            intuition=(
                "Learn a general starting point across many small jobs, then "
                "quickly retune on a handful of examples from a new job."
            ),
            formal_idea=(
                "θ₀ = fit(∪_τ train_τ); θ'_τ = fit_from(θ₀, S_τ)."
            ),
            why_it_matters=(
                "Gives a practical transfer/init path without claiming full MAML.",
            ),
            how_buildml_uses=(
                "session.metalearning.fit(method='warm_start', base_estimator=...); session.metalearning.adapt.",
            ),
            interpretation_rules=(
                "Compare warm_start vs prototypical episodic mean_accuracy on holdout.",
            ),
            assumptions=("Shared label space across tasks (global LabelEncoder).",),
            failure_modes=("Support too small for the base estimator to refit.",),
            anti_patterns=("Calling warm-start 'MAML' or second-order meta-gradients.",),
            worked_example_pattern=(
                "session.metalearning.fit(method='warm_start') → session.metalearning.adapt(task_id=...).",
            ),
            related_concepts=("metalearning-episodic", "metalearning-prototypical"),
        ),
        _note(
            key="metalearning-torch-prototypical",
            title="Torch tabular prototypical (deep encoder)",
            summary="MLP encoder + embedding-space nearest prototypes for few-shot tabular tasks.",
            definition=(
                "method='prototypical_torch' with backend='torch' trains a small MLP "
                "encoder with episodic prototype cross-entropy, then adapts via "
                "embedding-space class means."
            ),
            intuition=(
                "Learn a shared sketch space across tasks, then few-shot classify "
                "by nearest class sketch."
            ),
            formal_idea=(
                "φ_θ = MLP; c_k = mean({φ_θ(x) : x ∈ S, y=k}); "
                "ŷ = argmin_k ||φ_θ(x) − c_k||₂."
            ),
            why_it_matters=(
                "Honest deep few-shot baseline without vision ProtoNet claims.",
            ),
            how_buildml_uses=(
                "session.metalearning.fit(backend='torch', method='prototypical_torch').",
            ),
            interpretation_rules=(
                "Compare meta_train_accuracy and holdout mean_accuracy vs sklearn prototypical.",
            ),
            assumptions=("buildml[torch] installed; numeric tabular features.",),
            failure_modes=("Torch missing; insufficient episodic rows.",),
            anti_patterns=("Claiming image ProtoNet or foundation-model meta-learning.",),
            worked_example_pattern=(
                "session.metalearning.fit(backend='torch', method='prototypical_torch', k_shot=5)."
            ,),
            related_concepts=("metalearning-episodic", "metalearning-prototypical", "metalearning-maml"),
        ),
        _note(
            key="metalearning-maml",
            title="Industry tabular MAML/Reptile (first-order)",
            summary="Inner-loop support adapt + outer episodic meta-update: small-scale, not MAML-at-scale.",
            definition=(
                "backend='industry' with method='maml' or 'reptile' runs first-order "
                "tabular meta-learning with optional learn2learn MAML wrapper."
            ),
            intuition=(
                "Practice quick retunes on many small jobs so the starting weights "
                "help on the next job's few labels."
            ),
            formal_idea=(
                "θ' = θ − α∇_θ L_S(θ); meta-update from query loss or Reptile "
                "interpolation toward θ'."
            ),
            why_it_matters=(
                "Provides honest task-adaptation beyond warm-start without "
                "second-order MAML-at-scale claims."
            ,),
            how_buildml_uses=(
                "session.metalearning.fit(backend='industry', method='maml'); session.metalearning.adapt.",
            ),
            interpretation_rules=(
                "Read inner_steps, meta_train_accuracy, novel_task_ids on eval.",
            ),
            assumptions=("Torch available; shared classification label space.",),
            failure_modes=("Support too small for inner loop; torch missing.",),
            anti_patterns=(
                "Equating with EconML causal meta or full second-order MAML papers.",
            ),
            worked_example_pattern=(
                "session.metalearning.fit(backend='industry', method='maml', inner_steps=5)."
            ,),
            related_concepts=("metalearning-episodic", "metalearning-warm-start"),
        ),
        _note(
            key="metalearning-bundle-boundary",
            title="Meta-learning bundle boundary",
            summary="buildml.metalearning_bundle.v1 stores MetaLearningPlan; Session checkpoints do not embed it.",
            definition=(
                "A meta-learning bundle persists the episodic protocol, feature/task "
                "contract, label encoder, and optional warm-start init estimator. "
                "Session checkpoints persist data/roles/splits/history: not "
                "MetaLearningPlan."
            ),
            intuition=(
                "Saving the binder resume is not the same as saving the few-shot "
                "meta-learner."
            ),
            formal_idea=(
                "Artifacts are complementary: checkpoint_load ↛ meta-learner; "
                "session.metalearning.load_bundle ↛ dataset rows."
            ),
            why_it_matters=("Mixing artifacts causes silent missing-learner failures.",),
            how_buildml_uses=("session.metalearning.save_bundle / session.metalearning.load_bundle.",),
            interpretation_rules=("Read meta.json format buildml.metalearning_bundle.v1.",),
            assumptions=("Feature/task/target columns still match at load time.",),
            failure_modes=("Expecting checkpoint_load to restore MetaLearningPlan.",),
            anti_patterns=("Treating multitask or online bundles as meta-learning plans.",),
            worked_example_pattern=(
                "session.metalearning.save_bundle(path); other.metalearning.load_bundle(path).",
            ),
            related_concepts=("metalearning-episodic", "multitask-bundle-boundary"),
        ),
    )
}
