# ruff: noqa: E501
"""Dl concept notes."""

from __future__ import annotations

from buildml.explain.concepts._builder import _note
from buildml.explain.schemas import ConceptNote

DL_NOTES: dict[str, ConceptNote] = {
    note.key: note
    for note in (
        _note(
            key="batch-leakage",
            title="Batch and loader leakage",
            summary="Train-only shuffling and train-fit batch transforms must not remix evaluation rows into learning.",
            definition=(
                "Batch leakage occurs when evaluation-partition rows influence training batches—through "
                "shared shuffling, oversampling, or statistics (normalize/augment) fit on more than train."
            ),
            intuition=(
                "If the DataLoader that updates weights can see test rows, or if batch normalize peeks at "
                "holdout values, the network practices on the exam."
            ),
            formal_idea=(
                "Let partitions be disjoint. A train DataLoader may shuffle within train only. Any transform "
                "parameters θ_batch = L(train) apply frozen to validation/test loaders."
            ),
            why_it_matters=(
                "Loader mistakes create optimistic Torch metrics that classical split discipline alone cannot catch.",
                "Normalize fit on all partitions is the neural analogue of scaling before train_test_split.",
            ),
            how_buildml_uses=(
                "Session.make_torch_loaders shuffles the train loader only.",
                "Optional standardize fits mean/std on train and freezes them on validation/test.",
                "Catalog leakage notes call out shuffle and normalize scope for Torch ops.",
            ),
            interpretation_rules=(
                "If shuffle was enabled on validation/test loaders, treat subsequent scores as contaminated.",
                "Empty holdout loaders are a data issue, not a reason to merge partitions.",
            ),
            assumptions=(
                "Split membership is defined before loader construction.",
                "Feature matrices are prepared without refitting on evaluation rows.",
            ),
            failure_modes=(
                "Concatenating partitions into one Dataset with a single shuffle flag.",
                "Global StandardScaler fit before building partition loaders.",
            ),
            anti_patterns=(
                "Building one shuffled DataLoader over the full table, then slicing batches by index later.",
            ),
            worked_example_pattern=(
                "Split → make_torch_loaders(shuffle_train=True) → assert validation/test loaders do not shuffle.",
                "Compare train-fit normalize versus full-table normalize on the same holdout.",
            ),
            related_concepts=("leakage-boundary", "evaluation-partitions", "data-splitting"),
        ),
        _note(
            key="early-stopping-partition",
            title="Early-stopping partition",
            summary="Stopping rules may read validation metrics; the test partition remains a final estimate only.",
            definition=(
                "Early stopping selects a training epoch using a monitor partition—almost always validation—"
                "so that test metrics stay out of the stopping decision."
            ),
            intuition=(
                "Validation tells you when to put the pencil down. If you watch the official test score to "
                "decide when to stop, the official score is no longer independent."
            ),
            formal_idea=(
                "Choose epoch t* = argmin_t M(validation_t). Report generalization with M(test_{t*}) only after "
                "t* is fixed. Using M(test) inside the argmin biases the reported score."
            ),
            why_it_matters=(
                "Neural nets overfit easily; stopping on test hides that overfitting.",
                "Teaching and model cards need the monitor partition named beside the selected epoch.",
            ),
            how_buildml_uses=(
                "fit_torch early_stopping_patience monitors validation (default monitor=val_loss).",
                "TrainResult.early_stop records triggered/best_epoch/reason and restore_best_weights.",
                "evaluate_torch defaults to partition='test' for final scoring after training choices freeze.",
                "Catalog anti-patterns warn against test-tuned stopping.",
            ),
            interpretation_rules=(
                "Read every curve with its partition tag.",
                "If stopping used test, treat the test metric as optimistic.",
            ),
            assumptions=(
                "A validation partition exists when early stopping will be enabled.",
                "Train/val/test membership stays fixed across the run.",
            ),
            failure_modes=(
                "Selecting the best test epoch after the fact and reporting that test score.",
                "Retuning patience repeatedly against the same test split.",
            ),
            anti_patterns=(
                "Using test loss as the early-stopping monitor.",
            ),
            worked_example_pattern=(
                "fit_torch(..., early_stopping_patience=3) → read early_stop.reason → "
                "evaluate_torch(partition='test') once.",
            ),
            related_concepts=("evaluation-partitions", "leakage-boundary", "batch-leakage", "training-curves"),
        ),
        _note(
            key="training-curves",
            title="Training curves",
            summary="Epoch loss/metric trajectories need device, monitor partition, and honesty limits beside the plot.",
            definition=(
                "A training curve is the time series of train (and optional validation) losses or "
                "metrics across epochs, optionally with learning-rate steps from a scheduler."
            ),
            intuition=(
                "Curves show whether the network is still learning, plateauing, or memorizing. "
                "Without naming the validation monitor and device, the picture is incomplete."
            ),
            formal_idea=(
                "For epochs t=1..T, record L_train(t) and optionally L_val(t). Early stopping "
                "selects t* from L_val. Claims about generalization require a separate held-out "
                "estimate after t* is fixed."
            ),
            why_it_matters=(
                "Batch losses are noisy; epoch aggregates are the teaching default.",
                "Validation improvement is not a test result; curves alone do not prove deployment risk.",
            ),
            how_buildml_uses=(
                "TrainResult.history and TrainingCurveReport store epoch series plus disclosures.",
                "Session.torch_training_curve() and walkthrough torch_training_status surface limits.",
                "Teaching Studio cockpit discloses early-stop partition and resolved device when a trainer exists.",
            ),
            interpretation_rules=(
                "Prefer epoch aggregates over batch spikes when comparing runs.",
                "If train falls while validation rises, treat later epochs as overfitting risk.",
                "Read early_stop.partition before quoting a selected epoch.",
            ),
            assumptions=(
                "History was logged under a fixed split and feature contract.",
                "Scheduler and clipping settings are part of the run identity.",
            ),
            failure_modes=(
                "Comparing curves from different devices or normalize contracts without disclosure.",
                "Reading resume-appended history as a single uninterrupted LR schedule when the scheduler changed.",
            ),
            anti_patterns=(
                "Publishing a loss plot without stating validation vs test scope.",
            ),
            worked_example_pattern=(
                "fit_torch → torch_training_curve → read disclosures → evaluate_torch(partition='test').",
            ),
            related_concepts=("early-stopping-partition", "evaluation-partitions", "batch-leakage"),
        ),
    )
}

