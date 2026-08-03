# ruff: noqa: E501
"""Beginner layers for episodic meta-learning."""

from __future__ import annotations

from buildml.explain.beginner._builder import ADVANCED, CORE, BeginnerLayer, _index, _layer

METALEARNING_BEGINNER: dict[str, BeginnerLayer] = _index(
    _layer(
        "metalearning-episodic",
        plain=(
            "Meta-learning is learning how to learn quickly. Instead of one big model for one big dataset, "
            "you have many small related jobs: one per store, per client, per device: and you want a "
            "system that handles a brand-new job well after seeing only a handful of its labelled examples."
        ),
        analogy=(
            "A supply teacher who has taught dozens of classes. On the first morning with a new class they "
            "are not starting from nothing; they have learned what to look for in the first ten minutes."
        ),
        steps=(
            "Name the column that identifies which job each row belongs to: a `group` role or `task_column`.",
            "Meta-training uses only tasks from the training partition.",
            "Each round is an episode: a few labelled rows from one task are the *support* set, the rest are the *query* set.",
            "The system adapts using the support set, then is scored on the query set.",
            "Repeat over many episodes and many tasks, then evaluate on tasks the model never trained on.",
        ),
        use=(
            "When you have many small related datasets and too few rows in each to train separately.",
            "When new tasks appear regularly and you cannot retrain from scratch each time.",
        ),
        avoid=(
            "Do not use it when you have one dataset: that is ordinary supervised learning.",
            "Do not use it when the tasks are unrelated; there is no shared structure to transfer.",
        ),
        myths=(
            (
                "Meta-learning here means large-scale MAML on foundation models.",
                "This is small-scale tabular few-shot learning. It is honest and useful, and it is not the same as the research systems the name evokes.",
            ),
            (
                "Evaluating on the same task IDs as training is fine as long as the rows differ.",
                "Then you are measuring within-task generalization, not few-shot transfer to a new task. BuildML discloses task overlap for exactly this reason.",
            ),
        ),
        example=(
            "session.set_roles({'store_id': 'group', 'converted': 'target'})",
            "session.fit_metalearning(method='prototypical', k_shot=5, n_episodes=50)",
            "report = session.evaluate_metalearning(partition='validation')",
            "print(report.novel_task_ids, report.mean_accuracy)",
        ),
        check=(
            "How many distinct tasks are in your training partition? Fewer than a handful and there is nothing to generalize over.",
            "Do your evaluation tasks appear in training? If so, what you measured is not few-shot transfer.",
        ),
        tools=("fit_metalearning", "adapt_to_task", "evaluate_metalearning", "set_roles"),
        terms=("meta-learning", "few-shot", "support set", "leakage"),
        difficulty=ADVANCED,
    ),
    _layer(
        "metalearning-prototypical",
        plain=(
            "The prototypical method is the simplest thing that works. For each class, average the few "
            "labelled examples you have into one representative point: the prototype. Classify a new row "
            "by whichever prototype it sits closest to."
        ),
        analogy=(
            "Sketching the average face of each family from three photos, then deciding which family a new "
            "person belongs to by whichever sketch they most resemble."
        ),
        steps=(
            "Take the support set for a task: `k_shot` labelled rows per class.",
            "Average the feature vectors within each class to get one prototype per class.",
            "For a query row, measure the straight-line distance to every prototype.",
            "Assign the nearest one.",
            "That is the whole method: no weights are trained inside the episode.",
        ),
        use=(
            "As your first few-shot attempt; it is fast, has almost no knobs, and is a genuine baseline.",
            "When you have very few examples per class and a trained model would simply overfit.",
        ),
        avoid=(
            "Do not use it when a class is spread across several distinct clumps: one average point cannot represent two clusters.",
            "Do not use it on unscaled features; nearest-point logic is entirely at the mercy of column scale.",
        ),
        myths=(
            (
                "This is a neural prototypical network.",
                "It is nearest-centroid on your existing features. No embedding is learned. The torch variant is the version that learns one.",
            ),
            (
                "More shots always help.",
                "More shots give a steadier prototype, but they also mean fewer query rows and a less realistic few-shot test. The point is to work with few.",
            ),
        ),
        example=(
            "session.fit_metalearning(method='prototypical', k_shot=3, n_episodes=30)",
            "adapted = session.adapt_to_task(task_id='store_42')",
            "print(adapted.support_size, adapted.classes)",
        ),
        check=(
            "Are your features on comparable scales?",
            "Does every task have at least `k_shot` plus one row for each class?",
        ),
        tools=("fit_metalearning", "adapt_to_task", "evaluate_metalearning", "scale_features"),
        terms=("few-shot", "prototype", "support set", "feature scaling"),
        difficulty=CORE,
    ),
    _layer(
        "metalearning-warm-start",
        plain=(
            "Warm start trains one ordinary model on all your tasks pooled together, then uses it as a "
            "starting point. For a new task, it copies that model and refits it on the handful of examples "
            "you have: quicker and better than starting from nothing."
        ),
        analogy=(
            "Hiring someone with ten years in the industry rather than a new graduate. They still need a "
            "week to learn your specifics, but the week is enough."
        ),
        steps=(
            "Fit a logistic or SGD classifier on all training tasks pooled together: that is the meta-initialization.",
            "For a new task, clone it.",
            "Refit the clone on that task's small support set.",
            "Predict with the adapted clone.",
            "The pooled starting point carries what is common across tasks; the refit captures what is specific.",
        ),
        use=(
            "When tasks share a lot of structure and differ mainly in emphasis.",
            "When you want something that behaves like a normal sklearn model and is easy to reason about.",
        ),
        avoid=(
            "Do not use it if the label meanings differ between tasks: the pooled model assumes one shared label space.",
            "Do not use it when the support set is smaller than the base estimator needs to refit at all.",
        ),
        myths=(
            (
                "Warm start is MAML.",
                "MAML explicitly optimizes the initialization so that adaptation works well. Warm start just uses a pooled fit as the starting point. Similar shape, different guarantee.",
            ),
            (
                "Pooled pretraining always beats per-task training.",
                "When tasks conflict, the pooled model averages them into something that suits none. Compare against per-task fits before deciding.",
            ),
        ),
        example=(
            "session.fit_metalearning(",
            "    method='warm_start',",
            "    base_estimator=LogisticRegression(max_iter=1000),",
            ")",
            "session.adapt_to_task(task_id='client_new')",
        ),
        check=(
            "Does class 3 mean the same thing in every task?",
            "How does warm start compare with prototypical on your holdout tasks?",
        ),
        tools=("fit_metalearning", "adapt_to_task", "evaluate_metalearning"),
        terms=("meta-learning", "few-shot", "transfer learning", "support set"),
        difficulty=CORE,
    ),
    _layer(
        "metalearning-torch-prototypical",
        plain=(
            "This is the prototypical method with a learned representation. A small neural network learns "
            "to reposition your features into a space where the class averages separate cleanly, and then "
            "nearest-prototype classification happens in that new space."
        ),
        analogy=(
            "Rather than comparing photographs directly, first learn what to look at: jawline, eye "
            "spacing: and then compare on those. The comparison is easy once the right view is learned."
        ),
        steps=(
            "A small multilayer network encodes each row into an embedding.",
            "Training runs episodes: build prototypes in embedding space, score the query rows, adjust the network.",
            "Over many episodes the network learns an embedding where prototypes are far apart.",
            "For a new task, encode the support rows, average per class, classify the queries by nearest prototype.",
            "Requires `buildml[torch]`.",
        ),
        use=(
            "When plain nearest-centroid underperforms and you have enough tasks for the encoder to learn from.",
            "When the useful structure is a combination of columns rather than the raw columns themselves.",
        ),
        avoid=(
            "Do not use it with a handful of tasks; the encoder needs episodic variety or it just memorizes.",
            "Do not reach for it before checking that the plain prototypical baseline actually falls short.",
        ),
        myths=(
            (
                "Deep few-shot learning is always better than the simple version.",
                "On tabular data with modest task counts, nearest-centroid frequently wins. Measure both: that is why both exist.",
            ),
            (
                "This is the image prototypical network from the literature.",
                "Same idea, tabular encoder, much smaller scale. BuildML is explicit about this so nobody quotes vision-benchmark expectations.",
            ),
        ),
        example=(
            "# pip install \"buildml[torch]\"",
            "session.fit_metalearning(",
            "    backend='torch', method='prototypical_torch',",
            "    k_shot=5, n_episodes=200, random_state=0,",
            ")",
        ),
        check=(
            "Does it beat plain prototypical on your holdout tasks?",
            "How many distinct training tasks does your encoder get to see?",
        ),
        tools=("fit_metalearning", "adapt_to_task", "evaluate_metalearning"),
        terms=("few-shot", "embedding", "neural network", "extra"),
        difficulty=ADVANCED,
    ),
    _layer(
        "metalearning-maml",
        plain=(
            "MAML and Reptile go one step further than warm start: they deliberately shape the starting "
            "weights so that a few gradient steps on a new task's support set produce a good model. The "
            "initialization is optimized for adaptability rather than for accuracy on its own."
        ),
        analogy=(
            "Training an athlete not to be excellent at one sport but to be the kind of athlete who picks "
            "up any new sport in a week. You are optimizing for adaptability itself."
        ),
        steps=(
            "Inner loop: take a task, copy the weights, take a few gradient steps on its support set.",
            "Measure how well the adapted copy does on that task's query set.",
            "Outer loop: nudge the original weights so that future inner loops end up better.",
            "Reptile is the cheaper variant: just move the original weights toward the adapted ones.",
            "BuildML implements the first-order form, which is what makes it practical.",
        ),
        use=(
            "When warm start adapts too slowly and you have many training tasks.",
            "When each new task genuinely needs real adjustment, not just a nudge.",
        ),
        avoid=(
            "Do not use it with few tasks or tiny support sets; the inner loop needs something to work with.",
            "Do not use it if training time matters and warm start is already close.",
        ),
        myths=(
            (
                "This is full second-order MAML.",
                "BuildML runs the first-order approximation. It is far cheaper, usually close in practice, and labelled honestly.",
            ),
            (
                "Meta-learners here relate to the causal meta-learners in the causal module.",
                "Completely unrelated despite the shared word. Causal meta-learners estimate treatment effects; these adapt to new tasks.",
            ),
        ),
        example=(
            "session.fit_metalearning(",
            "    backend='industry', method='maml',",
            "    inner_steps=5, k_shot=5, random_state=0,",
            ")",
            "session.evaluate_metalearning(partition='test')",
        ),
        check=(
            "How many inner steps can your support set actually support?",
            "Is the gain over warm start worth the extra training time?",
        ),
        tools=("fit_metalearning", "adapt_to_task", "evaluate_metalearning"),
        terms=("meta-learning", "few-shot", "gradient descent", "extra"),
        difficulty=ADVANCED,
    ),
    _layer(
        "metalearning-bundle-boundary",
        plain=(
            "The meta-learning plan: the episodic protocol, the feature and task contract, the label "
            "encoder, and any warm-start initialization: saves as its own bundle. Session checkpoints do "
            "not carry it."
        ),
        analogy=(
            "The training regime and the trainee's current form are different records. Saving one does not "
            "save the other."
        ),
        steps=(
            "Fit a meta-learner so a plan exists.",
            "Call `save_metalearning_bundle(path)`.",
            "Reload with `load_metalearning_bundle(path)` in a Session holding the new task's rows.",
            "Call `adapt_to_task` to fit the few-shot adaptation for that specific task.",
            "Use checkpoints separately for data and workflow state.",
        ),
        use=(
            "When new tasks arrive in production and must be adapted without a full retrain.",
            "When the label encoding has to stay pinned so class 3 keeps meaning the same thing.",
        ),
        avoid=(
            "Do not confuse this with multitask or online bundles; they solve different problems and will not load here.",
            "Do not expect the bundle to hold a model for a specific task: it holds the machinery for producing one.",
        ),
        myths=(
            (
                "Loading the bundle gives you a ready-to-predict model.",
                "It gives you the meta-learner. You still call `adapt_to_task` with the new task's support rows before you can predict.",
            ),
            (
                "The task column can change after loading.",
                "The feature, task, and target contract is recorded and checked at load time. A mismatch is an error, deliberately.",
            ),
        ),
        example=(
            "session.save_metalearning_bundle('artifacts/store-fewshot')",
            "svc = Session.ingest(new_store_rows).load_metalearning_bundle('artifacts/store-fewshot')",
            "svc.adapt_to_task(task_id='store_new')",
        ),
        check=(
            "Do the new rows carry the same feature and task columns the bundle expects?",
            "Where does the support set for a brand-new task come from in production?",
        ),
        tools=("save_metalearning_bundle", "load_metalearning_bundle", "adapt_to_task", "checkpoint_save"),
        terms=("bundle", "checkpoint", "meta-learning", "few-shot"),
        difficulty=CORE,
    ),
)

__all__ = ["METALEARNING_BEGINNER"]
