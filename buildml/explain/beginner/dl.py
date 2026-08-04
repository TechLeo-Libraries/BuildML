# ruff: noqa: E501
"""Beginner layers for the deep-learning (Torch) surface."""

from __future__ import annotations

from buildml.explain.beginner._builder import CORE, BeginnerLayer, _index, _layer

DL_BEGINNER: dict[str, BeginnerLayer] = _index(
    _layer(
        "batch-leakage",
        plain=(
            "Neural networks train on small groups of rows called batches, served by a data loader. The "
            "loader is where leakage sneaks into deep learning: shuffling that mixes partitions, or a "
            "normalization computed across all rows, quietly lets evaluation data influence training."
        ),
        analogy=(
            "Shuffling the exam papers into the practice pile. Each individual sheet is fine; the mixing is "
            "the mistake."
        ),
        steps=(
            "Build one loader per partition, never a single loader over all rows.",
            "Shuffle only the training loader: shuffling evaluation loaders changes nothing useful and hides ordering bugs.",
            "Fit any batch-level transform (normalization statistics, augmentation parameters) on training rows only.",
            "Freeze those statistics and reuse them for validation and test batches.",
            "Confirm the row counts per loader match your split before you start a long run.",
        ),
        use=(
            "Every time you build Torch loaders from a Session, which is every deep-learning run.",
            "Especially with multimodal or text loaders, where the tokenizer or feature extractor may also be fitted from data.",
        ),
        avoid=(
            "Do not build loaders before the split exists: BuildML requires the boundary first, on purpose.",
            "Do not reuse a training loader for evaluation to 'save memory'; the shuffling and any train-time augmentation will corrupt the score.",
        ),
        myths=(
            (
                "Batch normalization statistics are harmless.",
                "In training mode they come from the batch; if evaluation batches update them, evaluation data has influenced the model. Freeze the model into eval mode.",
            ),
            (
                "Deep learning is too flexible to leak in the classical sense.",
                "It leaks in exactly the classical sense, plus new ways: loaders, augmentation, tokenizer vocabularies, and pretraining corpora that already contain your test rows.",
            ),
        ),
        example=(
            "session.split(test_size=0.2, validation_size=0.2, random_state=0)",
            "session.dl.make_loaders(batch_size=64, shuffle_train=True)",
            "session.dl.fit(epochs=20, monitor_partition='validation')",
            "session.dl.evaluate(partition='test')",
        ),
        check=(
            "Is your evaluation loader shuffled? It should not be.",
            "Where did your normalization statistics come from?",
        ),
        tools=("make_torch_loaders", "make_text_torch_loaders", "fit_torch", "evaluate_torch"),
        terms=("batch", "leakage", "epoch", "neural network", "split"),
        difficulty=CORE,
    ),
    _layer(
        "early-stopping-partition",
        plain=(
            "Early stopping watches a score during training and halts when it stops improving. Whichever "
            "partition it watches becomes part of the training process: so it must be validation, never "
            "test, or your final number is no longer independent."
        ),
        analogy=(
            "A coach deciding when you have practised enough by watching your mock exam. Perfectly sensible. "
            "Using the real exam for that purpose is not."
        ),
        steps=(
            "Pick the metric that matters and the partition to monitor: validation.",
            "Set patience: how many epochs without improvement you will tolerate before stopping.",
            "Train, letting the monitor decide when to stop and which epoch's weights to keep.",
            "Record the stopping epoch, because it is a fitted choice like any other hyperparameter.",
            "Score test once, after training has finished.",
        ),
        use=(
            "On essentially every neural-network run: it is the cheapest overfitting control you have.",
            "When training time is expensive and you would otherwise guess an epoch count.",
        ),
        avoid=(
            "Never monitor the test partition, even 'just to watch'. The stopping epoch is chosen from it, so it becomes selection data.",
            "Do not set patience to 1 on a noisy metric; you will stop on a random dip.",
        ),
        myths=(
            (
                "Early stopping on test is fine because the model does not train on those rows.",
                "The model does not fit those rows, but the *stopping point* is chosen using them. That is selection, and it inflates the final score.",
            ),
            (
                "The last epoch is the best model.",
                "With early stopping you keep the best-monitored epoch, which is usually several epochs before the run ended.",
            ),
        ),
        example=(
            "session.dl.fit(",
            "    epochs=100, early_stopping_patience=5,",
            "    monitor_partition='validation', monitor_metric='loss',",
            ")",
            "session.dl.evaluate(partition='test')",
        ),
        check=(
            "Which partition does your monitor read?",
            "At which epoch did training stop, and does the curve support that choice?",
        ),
        tools=("fit_torch", "torch_training_curve", "evaluate_torch", "cross_validate_torch"),
        terms=("early stopping", "epoch", "validation", "loss function", "overfitting"),
        difficulty=CORE,
    ),
    _layer(
        "training-curves",
        plain=(
            "A training curve plots loss or a metric against epoch, usually for both the training and the "
            "monitored partition. It is the single most informative picture in deep learning: but only if "
            "you also know which device it ran on, which partition it monitored, and what it cannot tell you."
        ),
        analogy=(
            "A heart-rate trace during exercise. Enormously informative to someone who knows whether you "
            "were running or sitting, and useless without that context."
        ),
        steps=(
            "Train while recording per-epoch loss and metrics for train and the monitored partition.",
            "Plot both lines together: the gap between them is the overfitting story.",
            "Read the shape: both still falling means undertrained; train falling while validation rises means overfitting; both flat and high means the model or the features are wrong.",
            "Note the device and the seed alongside the plot, because both change the trace.",
            "Do not read fine detail from a single run; epoch-level noise is large.",
        ),
        use=(
            "After every training run, before you look at any final metric.",
            "When deciding whether to train longer, change the learning rate, or change the architecture.",
        ),
        avoid=(
            "Do not compare curves from runs with different batch sizes or learning rates as if the x-axis meant the same thing.",
            "Do not plot the test partition on the curve: watching it is how it stops being a test partition.",
        ),
        myths=(
            (
                "A smooth falling curve means the model is good.",
                "It means optimization is working. A model can optimize beautifully toward a target you defined wrongly.",
            ),
            (
                "Validation loss rising always means stop.",
                "For some tasks validation loss rises while the metric you care about keeps improving, because the loss and the metric are not the same function.",
            ),
        ),
        example=(
            "session.dl.fit(epochs=50, monitor_partition='validation')",
            "curve = session.dl.training_curve()",
            "print(curve.epochs[-1], curve.train_loss[-1], curve.monitor_loss[-1])",
        ),
        check=(
            "Is the gap between your two lines growing, shrinking, or stable?",
            "Would the curve look the same with a different seed?",
        ),
        tools=("torch_training_curve", "fit_torch", "learning_curve", "evaluate_torch"),
        terms=("learning curve", "epoch", "loss function", "overfitting", "batch"),
        difficulty=CORE,
    ),
)

__all__ = ["DL_BEGINNER"]
