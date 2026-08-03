# ruff: noqa: E501
"""Plain-language glossary for every piece of jargon BuildML explanations use.

The explain system is layered so a complete beginner can read any note without
prior machine-learning vocabulary. That only works if the vocabulary itself is
resolvable in place, so this module is the single source of truth for what a
technical term *means in everyday words*.

Two things consume it:

* :mod:`buildml.explain.pedagogy` attaches an auto-detected glossary to every
  concept note and operation primer, so a reader never meets an undefined term.
* Authors reference entries by key when a note needs a specific term surfaced
  even though the exact string never appears in the prose.

Entries are intentionally short. A glossary entry is a bridge, not a lesson —
when a term deserves a lesson it also has a concept note, wired through
``CONCEPT_FOR_TERM``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from buildml.explain.schemas import GlossaryTerm


def _term(
    term: str,
    plain_meaning: str,
    *aliases: str,
) -> GlossaryTerm:
    return GlossaryTerm(term=term, plain_meaning=plain_meaning, also_called=tuple(aliases))


_ENTRIES: tuple[GlossaryTerm, ...] = (
    # ---- data and framing ------------------------------------------------
    _term("row", "One record in your table — one customer, one transaction, one day, one sensor reading. Machine learning almost always learns by comparing many rows.", "observation", "sample", "instance"),
    _term("column", "One measured property shared by every row, such as age or price.", "field", "variable", "attribute"),
    _term("feature", "A column the model is allowed to look at when making a prediction. Features are the inputs.", "predictor", "independent variable", "covariate"),
    _term("target", "The column you want the model to predict. It is the answer key during training.", "label", "outcome", "dependent variable", "response"),
    _term("role", "A label BuildML puts on a column saying how it may be used — feature, target, identifier, group, time, weight, or ignored. Roles matter more than the data type.", "column role"),
    _term("dtype", "The storage type of a column, such as integer or text. It says how the value is stored, not what it means.", "data type"),
    _term("cardinality", "How many distinct values a column has. 'High cardinality' means lots of unique values, like customer IDs.", "n_unique"),
    _term("categorical", "A column whose values are named groups rather than quantities, such as country or product type.", "nominal", "category column"),
    _term("numeric", "A column holding quantities you can meaningfully add or compare, such as price or age.", "continuous"),
    _term("ordinal", "A category with a real order, such as small / medium / large. The order carries meaning; the spacing usually does not.", "ordered category"),
    _term("missing value", "A blank cell. It may mean 'not measured', 'not applicable', or 'lost' — and those are different problems.", "null", "NaN", "NA"),
    _term("imputation", "Filling blank cells with a substitute value learned from the training rows, such as the median.", "impute", "fill-in"),
    _term("encoding", "Turning category names into numbers a model can consume, because most estimators only read numbers.", "encode"),
    _term("one-hot encoding", "Replacing one category column with one yes/no column per category, so no fake numeric order is invented.", "dummy variables", "one-hot"),
    _term("ordinal encoding", "Replacing categories with integers. Safe only when the categories genuinely have an order.", "label encoding"),
    _term("target encoding", "Replacing each category with the average target value seen for that category during training. Powerful and unusually easy to leak with.", "mean encoding", "likelihood encoding"),
    _term("scaling", "Rewriting numeric columns onto a common range so that a column measured in millions does not automatically dominate one measured in units.", "standardization", "normalization", "scale", "feature scaling"),
    _term("binning", "Replacing a numeric value with the named range it falls into, such as turning age into age bands.", "discretization", "bucketing"),
    _term("outlier", "A value far away from the rest. It might be an error, or it might be the most important row you have.", "extreme value"),
    _term("skew", "A lopsided distribution — most values bunched on one side with a long tail on the other.", "skewness"),
    _term("distribution", "The overall shape of the values in a column: what is common, what is rare, and how spread out they are."),
    _term("correlation", "A number between -1 and 1 saying how tightly two columns move together in a straight line. It never proves one causes the other."),
    _term("multicollinearity", "Several features carrying nearly the same information, which makes each one's individual contribution impossible to pin down.", "collinearity"),
    _term("dimensionality reduction", "Compressing many columns into a few summary columns that keep most of the variation.", "reduce dimensions"),
    _term("PCA", "Principal component analysis: it rotates your numeric columns into a small set of new columns ordered by how much variation they capture.", "principal component analysis", "principal components"),
    _term("embedding", "A row, word, image, or entity represented as a short list of numbers, arranged so that similar things end up close together.", "vector representation", "latent vector"),
    _term("feature engineering", "Building new columns from existing ones because the useful signal is easier for a model to see in the new form."),
    _term("feature selection", "Keeping a subset of columns and dropping the rest, usually to reduce noise, cost, or overfitting."),
    _term("design matrix", "The final all-numeric table handed to the estimator after every transform has run."),
    _term("schema", "The agreed list of column names, types, and meanings. Training and scoring must share one."),
    # ---- splitting, leakage, evaluation ----------------------------------
    _term("split", "Dividing rows into groups — typically train, validation, and test — so you can measure performance on rows the model never learned from.", "partition", "data split"),
    _term("train", "The rows the model is allowed to learn from. Every learned statistic must come from here.", "training set", "training partition"),
    _term("validation", "The rows used to compare options — which model, which settings, which threshold — while you are still deciding.", "dev set", "validation partition"),
    _term("test", "The rows held back untouched until every choice is locked, used once to estimate real-world performance.", "test partition"),
    _term("holdout", "Any rows deliberately kept out of training so they can give an honest score. Validation and test rows are both holdouts; 'test' is the stricter one you only look at once."),
    _term("leakage", "When information the model could not possibly have at prediction time sneaks into training. It produces great scores and terrible reality.", "data leakage", "target leakage"),
    _term("stratified", "Splitting in a way that keeps the mix of classes roughly the same in every partition.", "stratify"),
    _term("group split", "Splitting so that all rows belonging to the same person, device, or account stay on the same side of the boundary."),
    _term("time split", "Splitting by date so training always comes before evaluation, mirroring real forecasting.", "chronological split"),
    _term("cross-validation", "Repeatedly splitting the training rows into folds, training on most and scoring on the rest, so you see how much the score wobbles.", "CV", "k-fold"),
    _term("fold", "One of the equal slices cross-validation cuts the training rows into."),
    _term("out-of-fold", "A prediction made for a row by a model that was not trained on that row. It is the honest way to build stacked features.", "OOF"),
    _term("nested cross-validation", "Cross-validation inside cross-validation: the inner loop picks settings, the outer loop scores the whole selection procedure.", "nested CV"),
    _term("overfitting", "Memorizing quirks of the training rows instead of learning the pattern, so performance collapses on new data.", "overfit"),
    _term("underfitting", "A model too simple to capture the real pattern, so it does poorly everywhere including on training rows.", "underfit"),
    _term("generalization", "How well a model performs on rows it has never seen. It is the only performance that matters."),
    _term("bias-variance tradeoff", "The tension between a model too rigid to fit the truth and a model so flexible it fits the noise."),
    _term("baseline", "The dumbest reasonable prediction — always guessing the most common class, or always guessing the average. If your model cannot beat it, it is not adding value."),
    _term("metric", "A single number summarizing how good predictions are. Every metric hides something, so always read it with its partition and baseline.", "score"),
    _term("accuracy", "The share of predictions that were correct. Misleading when one class is rare: predicting 'no' always can score 99%."),
    _term("precision", "Of the rows the model flagged, how many were genuinely positive. It answers 'when it says yes, can I trust it?'"),
    _term("recall", "Of the genuinely positive rows, how many the model caught. It answers 'how much did we miss?'", "sensitivity", "true positive rate"),
    _term("F1", "The balanced blend of precision and recall, useful when you care about both and the positive class is rare.", "F1 score", "f1"),
    _term("ROC-AUC", "The chance that a randomly chosen positive row scores higher than a randomly chosen negative one. Ranking quality, not calibration.", "AUC", "roc_auc", "AUROC"),
    _term("PR-AUC", "Area under the precision-recall curve. Far more informative than ROC-AUC when positives are rare.", "average precision", "AP"),
    _term("confusion matrix", "A small table counting correct and incorrect predictions per class, so you can see *which* mistakes are being made."),
    _term("MAE", "Mean absolute error: the average size of the miss, in the same units as the target.", "mean absolute error"),
    _term("RMSE", "Root mean squared error: like MAE but punishes big misses much harder.", "root mean squared error"),
    _term("MAPE", "Mean absolute percentage error: the average miss expressed as a percentage. It explodes when actual values are near zero.", "mean absolute percentage error"),
    _term("R2", "The share of the target's variation the model explains. Zero means no better than predicting the average.", "R-squared", "r2"),
    _term("threshold", "The cutoff that turns a probability into a decision. Moving it trades false alarms against misses; it is a business choice, not a modeling one.", "cutoff", "operating point"),
    _term("calibration", "Whether a predicted 70% actually happens about 70% of the time. A model can rank perfectly and still be badly calibrated.", "calibrated probabilities"),
    _term("class imbalance", "One outcome being far rarer than the other, which breaks accuracy and default thresholds.", "imbalanced"),
    _term("resampling", "Changing how often each class appears in the *training* rows, by duplicating rare rows or dropping common ones.", "oversampling", "undersampling", "SMOTE"),
    _term("drift", "The data changing over time or between populations, so a model trained on yesterday no longer matches today.", "dataset drift", "distribution shift"),
    _term("error slice", "A subgroup of rows — a region, a channel, an age band — where the model is much worse than average.", "slice"),
    _term("confidence interval", "A range that plausibly contains the true value, given how much data you have. It is how you avoid over-reading a single number."),
    _term("statistical significance", "Evidence that an observed difference is unlikely to be pure chance. It does not mean the difference is large or useful."),
    _term("p-value", "How surprising your data would be if nothing were really going on. Small means surprising; it is not the probability that you are right."),
    # ---- models and training ---------------------------------------------
    _term("model", "A rule learned from data that turns feature values into a prediction.", "learner"),
    _term("estimator", "The scikit-learn term for an object that can be fitted to data and then make predictions. In practice it is the same thing as a model, named after the interface rather than the idea."),
    _term("fit", "Training: showing the estimator the training rows so it can learn its internal numbers.", "fitting"),
    _term("predict", "Applying a fitted model to new rows to get outputs.", "inference", "scoring"),
    _term("predict_proba", "Asking a classifier for the probability of each class rather than a single hard answer.", "probability output"),
    _term("hyperparameter", "A setting you choose *before* training, like tree depth or learning rate, as opposed to something the model learns.", "hyperparameters", "tuning parameter"),
    _term("parameter", "A number the model learns during training, such as a regression coefficient."),
    _term("grid search", "Trying every combination in a list of candidate settings and keeping the best-scoring one.", "GridSearchCV"),
    _term("randomized search", "Sampling random combinations of settings instead of trying all of them — usually much cheaper for similar quality.", "RandomizedSearchCV"),
    _term("Bayesian optimization", "Search that learns from earlier trials to decide which settings to try next, instead of guessing blindly.", "Optuna", "TPE"),
    _term("regularization", "Deliberately penalizing complexity so the model prefers simpler explanations and overfits less.", "L1", "L2", "penalty", "ridge", "lasso"),
    _term("linear model", "A model that adds up weighted feature values. Fast, interpretable, and blind to interactions you do not build for it.", "linear regression", "logistic regression"),
    _term("decision tree", "A model that asks a sequence of yes/no questions about feature values and reads off an answer at the leaf.", "tree"),
    _term("random forest", "Many decision trees trained on different random slices, whose votes are averaged to cancel out individual mistakes.", "RandomForest"),
    _term("gradient boosting", "Trees trained one after another, where each new tree focuses on the errors the previous ones made.", "GBDT", "XGBoost", "LightGBM", "CatBoost", "boosting"),
    _term("ensemble", "Combining several models so their independent errors partly cancel out.", "ensembling"),
    _term("bagging", "Training the same kind of model on many random samples of the data and averaging them, to reduce variance."),
    _term("voting", "Combining different model types by averaging their probabilities or taking a majority vote."),
    _term("stacking", "Training a small extra model to learn how much to trust each base model's prediction.", "stacked generalization", "meta-learner"),
    _term("blending", "Like stacking, but the combiner is trained on a single held-out slice instead of cross-validated folds."),
    _term("AutoML", "Automated search over model families *and* preprocessing choices, rather than tuning one fixed model."),
    _term("pipeline", "Preprocessing steps and the model bundled into one object, so the exact same transforms run at training and scoring time."),
    _term("feature importance", "A ranking of which features the fitted model leaned on. It describes the model, not the world, and never proves causation."),
    _term("permutation importance", "Measuring how much the score drops when one column is randomly shuffled. Shared credit between correlated columns makes it noisy."),
    _term("SHAP", "A method that attributes a single prediction to each feature's contribution, with a consistent additive accounting.", "Shapley values"),
    _term("learning curve", "A plot of score versus training-set size, used to tell 'need more data' apart from 'need a better model'."),
    _term("seed", "A fixed number that makes randomness repeatable, so the same code gives the same result twice.", "random_state", "random seed"),
    _term("epoch", "One complete pass of the training data through a neural network."),
    _term("batch", "A small group of rows processed together during neural-network training.", "mini-batch", "batch size"),
    _term("learning rate", "How big a step training takes each update. Too large and it overshoots; too small and it crawls.", "lr"),
    _term("loss function", "The number training tries to make small. It defines what the model considers a mistake.", "loss", "objective", "criterion"),
    _term("gradient descent", "The optimization method that repeatedly nudges parameters in whichever direction reduces the loss.", "SGD", "optimizer", "Adam"),
    _term("neural network", "A model built from layers of simple numeric units, able to learn complex patterns given enough data.", "deep learning", "MLP", "neural net"),
    _term("early stopping", "Halting training as soon as validation performance stops improving, to avoid overfitting."),
    _term("dropout", "Randomly switching off part of a neural network during training so it cannot rely on any single path."),
    _term("backbone", "A large model pretrained on generic data, reused as a starting point for your specific task.", "pretrained model", "foundation model"),
    _term("fine-tuning", "Continuing to train a pretrained model on your own data so it adapts to your task.", "finetune"),
    _term("transfer learning", "Reusing what a model learned on one problem to get a head start on another."),
    _term("checkpoint", "A saved snapshot of workflow state — data, roles, split membership, history — so you can resume later."),
    _term("bundle", "A saved artifact holding a fitted plan plus the contract needed to reuse it correctly."),
    _term("serialization", "Writing an in-memory object to disk so it can be loaded back later.", "joblib", "pickle"),
    _term("reproducibility", "Being able to recreate a result exactly, which needs seeds, versions, inputs, and recorded choices — not just the code."),
    # ---- specialized domains ---------------------------------------------
    _term("clustering", "Grouping rows that resemble each other, without being told what the groups are.", "cluster", "k-means", "DBSCAN"),
    _term("centroid", "The average position of a cluster — its center of mass."),
    _term("silhouette", "A score from -1 to 1 saying how much better each row fits its own cluster than the nearest other one. Geometry, not truth.", "silhouette score"),
    _term("unsupervised", "Learning structure from data with no target column to check against.", "unsupervised learning"),
    _term("supervised", "Learning from examples that already carry the right answer.", "supervised learning"),
    _term("semi-supervised", "Learning from a small labeled set plus a large unlabeled one.", "semisupervised", "label propagation", "self-training"),
    _term("self-supervised", "Creating a training signal from the data itself — hide part of the input and predict it — so no human labels are needed.", "SSL", "pretext task"),
    _term("active learning", "Letting the model choose which unlabeled rows a human should label next, to spend labeling effort where it helps most."),
    _term("pseudo-label", "A model-generated label used as if it were real. Useful and dangerous in equal measure."),
    _term("online learning", "Updating a model incrementally as new data arrives, instead of retraining from scratch.", "partial_fit", "incremental learning", "continual learning"),
    _term("anomaly detection", "Finding rows that do not look like the rest, usually without labeled examples of what 'bad' looks like.", "outlier detection", "novelty detection"),
    _term("alert rate", "The fraction of rows your threshold flags. It sets how much human review work you are creating."),
    _term("forecasting", "Predicting future values of a series from its own past, where order and timing matter.", "time series forecasting"),
    _term("lag feature", "The value of a column some number of steps earlier, turned into a column so ordinary models can use history.", "lag"),
    _term("horizon", "How many steps into the future you are predicting."),
    _term("seasonality", "A pattern that repeats on a fixed cycle, like weekday effects or December spikes."),
    _term("exogenous", "An outside driver used to help a forecast, such as price or weather. Using it for the future requires knowing its future values.", "exog"),
    _term("stationarity", "A series whose statistical behavior does not drift over time. Many classical forecasting methods assume it."),
    _term("causal inference", "Estimating what *would* happen if you intervened, which is a strictly stronger claim than noticing a correlation.", "causal"),
    _term("treatment", "The intervention whose effect you want to measure — the coupon sent, the drug given, the change shipped."),
    _term("confounder", "Something that influences both the treatment and the outcome, faking an effect if you ignore it.", "confounding"),
    _term("ATE", "Average treatment effect: how much the outcome changes on average if everyone got the treatment versus nobody.", "average treatment effect"),
    _term("propensity", "The estimated probability that a row received the treatment, given its features.", "propensity score"),
    _term("IPW", "Inverse propensity weighting: re-weighting rows so treated and untreated groups become comparable.", "inverse propensity weighting"),
    _term("doubly robust", "An estimator that stays correct if *either* the outcome model or the treatment model is right, not necessarily both.", "AIPW"),
    _term("unconfoundedness", "The assumption that you measured every common cause of treatment and outcome. It is an assumption, never a result.", "ignorability"),
    _term("positivity", "The assumption that every kind of row had some real chance of getting either treatment.", "overlap"),
    _term("graph", "Data described as nodes joined by edges — users and friendships, papers and citations, accounts and transfers.", "network"),
    _term("node", "One entity in a graph — a person, an account, a page. Where a table would give it a row, a graph gives it a point with connections.", "vertex"),
    _term("edge", "One connection between two nodes in a graph, such as 'follows', 'paid', or 'links to'. The connections are the data, not decoration.", "link"),
    _term("GNN", "Graph neural network: a model that lets each node's prediction depend on its neighbors.", "graph neural network", "GCN", "GraphSAGE", "GAT"),
    _term("transductive", "Training with the whole graph structure visible, including the nodes you will predict — only their labels are hidden."),
    _term("inductive", "Training on a subgraph only, so the model must generalize to nodes it never saw."),
    _term("PageRank", "A score of node importance based on how many important nodes point at it."),
    _term("knowledge graph", "Facts stored as (head, relation, tail) triples, such as (Paris, capital_of, France).", "KG", "triples"),
    _term("link prediction", "Guessing which missing connections in a graph are probably real."),
    _term("MRR", "Mean reciprocal rank: how high up the correct answer usually appears in a ranked list.", "mean reciprocal rank"),
    _term("nDCG", "A ranking score that rewards putting highly relevant items near the top, discounted by position.", "NDCG", "discounted cumulative gain"),
    _term("learning to rank", "Training a model to order items within a query rather than to score them independently.", "LTR", "ranker"),
    _term("recommender", "A system that suggests items to users based on past interactions or item content.", "recommendation", "collaborative filtering"),
    _term("cold start", "The problem of serving a user or item you have no history for."),
    _term("matrix factorization", "Discovering hidden user and item traits by factoring the interaction table into two smaller ones.", "ALS", "BPR", "latent factors"),
    _term("implicit feedback", "Signals like clicks and views, where you see what happened but never see an explicit rating or a confirmed 'no'."),
    _term("reinforcement learning", "Learning by trying actions and observing rewards, rather than from labeled examples.", "RL"),
    _term("agent", "The decision-maker in reinforcement learning — the thing choosing actions."),
    _term("policy", "The rule mapping a situation to an action — the agent's strategy, and the thing reinforcement learning is trying to improve."),
    _term("reward", "The numeric feedback telling the agent how good an outcome was."),
    _term("episode", "One complete run from a starting situation to an end — a whole game, a whole session, a whole journey. Reinforcement learning measures progress in episodes."),
    _term("policy gradient", "A family of reinforcement-learning methods that nudge the policy directly: make the actions that led to good outcomes a bit more likely, and the rest a bit less. REINFORCE is the simplest one.", "REINFORCE", "gradient"),
    _term("bandit", "The simplest reinforcement-learning setting: pick one option, see one reward, no long-term state to track.", "contextual bandit", "LinUCB"),
    _term("exploration", "Deliberately trying options you are unsure about, so you can learn instead of repeating a possibly wrong favorite.", "exploration-exploitation", "epsilon-greedy"),
    _term("off-policy evaluation", "Estimating how a new policy would have done, using logs collected under a different policy.", "OPE", "IPS", "counterfactual evaluation"),
    _term("imitation learning", "Learning a policy by copying recorded expert behavior instead of exploring.", "behavioral cloning", "BC"),
    _term("meta-learning", "Learning how to learn: training across many small tasks so a new task needs only a handful of examples.", "few-shot", "MAML"),
    _term("support set", "The handful of labeled examples given for a new task in few-shot learning."),
    _term("query set", "The examples used to score a few-shot model after it has seen the support set."),
    _term("prototype", "The average embedding of a class's support examples, used as that class's stand-in."),
    _term("federated learning", "Training across separate data holders by exchanging model updates instead of raw data.", "FedAvg", "FedProx"),
    _term("multi-task", "One model trained to predict several targets at once, sharing what it learns between them.", "multitask", "multi-output"),
    _term("probabilistic model", "A model that outputs a distribution — a value plus honest uncertainty — instead of a bare number.", "Bayesian"),
    _term("prediction interval", "A range that should contain the true value a stated share of the time, such as 90%."),
    _term("conformal prediction", "A distribution-free way to build prediction intervals with a guaranteed coverage rate, calibrated on held-out residuals.", "conformal"),
    _term("Gaussian process", "A flexible model that predicts a mean and a spread, and gets appropriately unsure far from the data it saw.", "GP"),
    _term("symbolic AI", "Explicit human-readable if-then rules over columns, rather than learned numeric weights.", "rule-based", "rules"),
    _term("neuro-symbolic", "Combining a learned statistical model with explicit rules that constrain or correct it."),
    _term("case-based reasoning", "Solving a new case by retrieving the most similar past cases and reusing their outcomes.", "CBR"),
    _term("topological data analysis", "Measuring the shape of data — loops, clusters, voids — in a way that survives stretching and noise.", "TDA", "persistent homology"),
    _term("persistence diagram", "A summary of which shape features appear and disappear as you zoom out on a point cloud."),
    _term("synthetic data", "Artificial rows generated to resemble real ones, used for augmentation, sharing, or stress testing."),
    _term("TSTR", "Train on synthetic, test on real: the honest way to check whether synthetic data is actually useful.", "train-synthetic-test-real"),
    _term("differential privacy", "A formal guarantee that a released result barely changes whether or not any single person is in the data."),
    _term("RAG", "Retrieval-augmented generation: search your documents first, then ask a language model to answer using only what was retrieved.", "retrieval-augmented generation"),
    _term("chunk", "A document cut into a passage small enough to embed and retrieve on its own."),
    _term("vector index", "A structure that finds the stored embeddings closest to a query embedding, fast.", "vector store", "FAISS", "ANN"),
    _term("recall@k", "Of the passages that should have been found, the share that appeared in the top k results."),
    _term("prompt injection", "Text hidden inside your data that tries to hijack a language model's instructions."),
    _term("LLM", "Large language model: a model that generates text and can follow instructions.", "large language model"),
    _term("token", "The chunk of text a language model actually reads — roughly a short word or word-piece."),
    _term("hallucination", "A confidently stated but fabricated answer from a language model."),
    _term("optimization", "Choosing the best action under constraints, once you already have predictions.", "decision optimization", "linear programming", "knapsack"),
    _term("cost matrix", "A table saying what each kind of mistake costs, so the model can minimize money rather than error count."),
    _term("expected value", "The average outcome, weighting each possibility by how likely it is."),
    # ---- similarity, geometry, and shared modelling vocabulary -----------
    _term("nearest neighbours", "The rows closest to yours under some distance measure. Many methods answer a question by looking at them and nothing else.", "nearest neighbors", "kNN", "k-nearest neighbours"),
    _term("distance metric", "The rule for measuring how far apart two rows are — straight-line, block-by-block, or angle-based. Change the rule and 'similar' changes with it.", "euclidean", "cosine similarity"),
    _term("coefficient", "The weight a linear model attaches to one feature: how much the prediction moves when that feature moves by one unit.", "weight", "coef"),
    _term("attribution", "Splitting one prediction into per-input contributions so you can say which inputs pushed it which way.", "contribution"),
    _term("curse of dimensionality", "The problem that adding columns or bins multiplies the space to be covered, so your data becomes sparse and distances stop being informative."),
    _term("privacy", "Whether a released model or dataset can reveal something about an individual who was in the training data. Separate from, and not implied by, accuracy or realism."),
    _term("contamination", "Information from your evaluation data reaching training — often through duplicate rows or documents — which makes holdout scores optimistic.", "duplicate leakage"),
    _term("aggregation", "Combining many partial results into one — averaging client model weights, pooling per-group scores, merging round updates.", "averaging"),
    _term("client", "One data holder in a federated setting: a hospital, a branch, a device. Its rows stay on its own side of the boundary.", "site", "participant"),
    _term("backend", "Which underlying library or implementation BuildML routes an operation to. Same call, different engine, sometimes different available options.", "engine backend"),
    # ---- text and natural language ---------------------------------------
    _term("corpus", "A collection of documents treated as one body of text to model or search.", "document collection"),
    _term("tokenization", "Splitting text into the individual units — usually words — a model will count or read.", "tokenize", "tokenizer"),
    _term("stopword", "A very common word such as 'the' or 'and' that many text methods discard because it appears everywhere.", "stop word", "stopwords"),
    _term("stemming", "Chopping words back to a rough common root so 'running' and 'runs' count as the same token.", "lemmatization", "stem"),
    _term("vocabulary", "The set of words a fitted text model knows about, learned from the training documents.", "vocab"),
    _term("out-of-vocabulary", "A word in new text that the fitted vocabulary never saw, so the model simply cannot see it.", "OOV", "out of vocabulary"),
    _term("n-gram", "A run of n adjacent units — 'credit card' is a word 2-gram; 'cred' is a character 4-gram.", "ngram", "bigram"),
    _term("TF-IDF", "A way of scoring words in a document: frequent here, rare across the corpus, therefore informative.", "tfidf", "term frequency-inverse document frequency"),
    _term("topic model", "An unsupervised method that finds recurring vocabulary patterns across a corpus and reports each as a ranked word list.", "LDA", "NMF", "topic modelling"),
    _term("coherence", "A score for whether a topic's top words genuinely appear together in real documents, rather than being an artifact.", "NPMI"),
    _term("keyphrase", "A short phrase that stands out in a document — frequent here, unusual elsewhere, or central in the document's word graph.", "key phrase", "keyword extraction"),
    _term("sentiment", "Whether text expresses a positive, negative, or neutral attitude.", "polarity", "valence"),
    _term("lexicon", "A dictionary of words with attached scores — sentiment values, categories — used by rule-based text methods.", "word list"),
    _term("named entity recognition", "Finding and typing the spans in text that name things: dates, amounts, people, organizations.", "NER", "entity extraction"),
    _term("gazetteer", "Your own list of known names or codes, matched literally in text so domain knowledge can enter without training data.", "term list"),
    _term("summarization", "Producing a shorter version of a document. Extractive summarization selects existing sentences; abstractive summarization writes new ones.", "summarisation"),
    _term("extractive", "Producing output made only of material copied verbatim from the input, so nothing can be invented.", "extractive summarization"),
    _term("language identification", "Working out which natural language a document is written in, so language-specific tooling can be chosen correctly.", "langid", "language detection"),
    _term("near-duplicate", "Two documents that are not byte-identical but are similar enough that treating them as independent evidence is wrong.", "near duplicate", "fuzzy duplicate"),
    _term("retrieval", "Finding the most relevant stored items for a query, by keyword match, by embedding similarity, or both.", "search", "retrieve"),
    _term("text features", "Turning a text column into ordinary numeric columns so a tabular model can use it alongside everything else.", "text feature expansion"),
    # ---- BuildML platform vocabulary -------------------------------------
    _term("Session", "The BuildML object that holds your data, roles, split, plans, and history, and exposes every operation."),
    _term("plan", "A fitted, frozen recipe BuildML stores — the imputer's medians, the cluster centroids, the encoder's vocabulary — so evaluation reuses training's exact decisions.", "fitted plan"),
    _term("history", "BuildML's ordered record of every operation you ran, with its parameters and the resulting state change."),
    _term("operation", "One public Session method, catalogued with its purpose, prerequisites, risks, and result reading.", "catalog operation"),
    _term("prerequisite", "Something that has to be true before an operation can run — data ingested, roles set, a split created."),
    _term("workflow resolution", "BuildML checking every catalogued operation against current state and marking it done, available, blocked, or skipped."),
    _term("dry run", "Previewing what an operation would do without changing anything."),
    _term("extra", "An optional dependency group you install with brackets, like ``pip install buildml[torch]``.", "optional dependency", "extras"),
    _term("disclosure", "A statement BuildML attaches to a result saying what it could not verify or had to approximate."),
    _term("provenance", "The traceable record of where data and results came from."),
)

GLOSSARY: dict[str, GlossaryTerm] = {entry.term.lower(): entry for entry in _ENTRIES}

_ALIAS_TO_KEY: dict[str, str] = {}
for _entry in _ENTRIES:
    _ALIAS_TO_KEY[_entry.term.lower()] = _entry.term.lower()
    for _alias in _entry.also_called:
        _ALIAS_TO_KEY.setdefault(_alias.lower(), _entry.term.lower())

# Terms that also have a full concept note, so the reader can go deeper.
CONCEPT_FOR_TERM: dict[str, str] = {
    "leakage": "leakage-boundary",
    "split": "data-splitting",
    "validation": "evaluation-partitions",
    "test": "evaluation-partitions",
    "holdout": "evaluation-partitions",
    "role": "column-roles",
    "schema": "feature-schema",
    "missing value": "missing-data",
    "imputation": "missing-data",
    "encoding": "categorical-encoding",
    "one-hot encoding": "categorical-encoding",
    "ordinal encoding": "categorical-encoding",
    "target encoding": "target-encoding",
    "scaling": "feature-scaling",
    "binning": "feature-binning",
    "outlier": "outlier-handling",
    "class imbalance": "class-imbalance",
    "resampling": "class-imbalance",
    "cross-validation": "cross-validation",
    "overfitting": "overfitting",
    "baseline": "baselines",
    "threshold": "thresholds",
    "calibration": "probability-calibration",
    "drift": "dataset-drift",
    "feature importance": "feature-importance",
    "permutation importance": "feature-importance",
    "PCA": "principal-components",
    "dimensionality reduction": "principal-components",
    "feature selection": "feature-selection",
    "seed": "reproducibility",
    "reproducibility": "reproducibility",
    "checkpoint": "checkpoint-integrity",
    "history": "operation-history",
    "dry run": "dry-run-plans",
    "multicollinearity": "variance-inflation",
    "clustering": "cluster-validity-not-truth",
    "silhouette": "cluster-validity-not-truth",
    "metric": "diagnostic-uncertainty",
    "confidence interval": "diagnostic-uncertainty",
    "ordinal": "categorical-encoding",
    "feature engineering": "encoding-imputation-scaling",
    "bias-variance tradeoff": "overfitting",
    "confusion matrix": "thresholds",
    "R2": "diagnostic-uncertainty",
    "parameter": "model-selection",
    "hyperparameter": "model-selection",
    "grid search": "automl-recipe-strategy-search",
    "randomized search": "automl-recipe-strategy-search",
    "Bayesian optimization": "automl-recipe-strategy-search",
    "ensemble": "ensemble-voting-vs-single-tree",
    "bagging": "ensemble-voting-vs-single-tree",
    "boosting": "ensemble-voting-vs-single-tree",
    "stacking": "ensemble-stacking-oof",
    "learning rate": "training-curves",
    "epoch": "training-curves",
    "early stopping": "early-stopping-partition",
    "stationarity": "forecast-lag-features",
    "query set": "metalearning-episodic",
    "support set": "metalearning-episodic",
    "hallucination": "nlp-vs-rag",
    "RAG": "nlp-vs-rag",
    "conformal prediction": "probabilistic-split-conformal",
    "anomaly": "anomaly-novelty-vs-unsupervised",
    "reward": "rl-tabular-q-learning",
    "policy": "rl-tabular-q-learning",
    "federated learning": "federated-fedavg",
    "differential privacy": "synthetic-privacy-limits",
    "cold start": "recommender-cold-start",
    "collaborative filtering": "recommender-collaborative-filtering",
    "tokenization": "nlp-text-normalization",
    "TF-IDF": "nlp-document-representation",
    "topic model": "nlp-topic-models",
    "named entity recognition": "nlp-rule-vs-statistical-ner",
    "prompt injection": "ai-prompt-injection",
}

_WORD_BOUNDARY = re.compile(r"\w")


def _pattern_for(text: str) -> re.Pattern[str]:
    escaped = re.escape(text)
    prefix = r"\b" if _WORD_BOUNDARY.match(text[0]) else ""
    suffix = r"\b" if _WORD_BOUNDARY.match(text[-1]) else ""
    return re.compile(prefix + escaped + suffix, re.IGNORECASE)


_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {}
for _key, _entry in GLOSSARY.items():
    _PATTERNS[_key] = tuple(
        _pattern_for(text) for text in (_entry.term, *_entry.also_called) if text
    )


def lookup(term: str) -> GlossaryTerm | None:
    """Find the definition of a term, whatever it is called.

    Vocabulary is not standardised across textbooks and libraries: the same idea
    is a 'holdout' in one place and a 'test partition' in another. Every entry
    therefore registers its aliases, and lookup is case-insensitive, so callers
    can pass whichever name the reader actually used.

    Parameters
    ----------
    term:
        The term or any of its recorded aliases. Leading and trailing
        whitespace and letter case are ignored.

    Returns
    -------
    ~buildml.explain.schemas.GlossaryTerm or None
        The entry, or ``None`` when BuildML has no definition for the term.

    See Also
    --------
    require : The same lookup, raising instead of returning ``None``.
    concept_for_term : The concept note that teaches a term in depth.
    """
    key = _ALIAS_TO_KEY.get(str(term).strip().lower())
    return GLOSSARY.get(key) if key else None


def require(term: str) -> GlossaryTerm:
    """Look a term up, refusing to continue when it is undefined.

    Use this where a missing definition is a bug rather than a possibility —
    for instance when a concept note declares the vocabulary it teaches, since a
    term named there but never defined would leave the reader stuck.

    Parameters
    ----------
    term:
        The term or any of its recorded aliases.

    Returns
    -------
    ~buildml.explain.schemas.GlossaryTerm
        The matching entry.

    Raises
    ------
    KeyError
        No entry matches the term or any alias.
    """
    entry = lookup(term)
    if entry is None:
        raise KeyError(f"Unknown BuildML glossary term: {term}")
    return entry


def detect_terms(
    texts: Iterable[str],
    *,
    limit: int | None = None,
    exclude: Iterable[str] = (),
) -> tuple[GlossaryTerm, ...]:
    """Find the jargon a passage uses, so it can be defined alongside it.

    This is what lets an explanation carry its own glossary rather than assuming
    the reader already knows the words in it. Matching is whole-word and
    case-insensitive, and results are ordered by first appearance so the reader
    meets each definition in the order the prose introduces it.

    Parameters
    ----------
    texts:
        The passages to scan. Empty entries are ignored.
    limit:
        Keep at most this many entries, dropping the ones that appear latest.
        ``None`` keeps all of them.
    exclude:
        Terms already defined elsewhere in the same answer, so a definition is
        not repeated. Matched case-insensitively against entry keys.

    Returns
    -------
    tuple of ~buildml.explain.schemas.GlossaryTerm
        Entries in order of first appearance in ``texts``.

    Examples
    --------
    >>> from buildml.explain.glossary import detect_terms
    >>> [entry.term for entry in detect_terms(["Fit on train, then check the holdout."])]
    ['fit', 'train', 'holdout']
    """
    excluded = {str(item).strip().lower() for item in exclude}
    blob = "\n".join(str(item) for item in texts if item)
    if not blob:
        return ()
    hits: list[tuple[int, str]] = []
    for key, patterns in _PATTERNS.items():
        if key in excluded:
            continue
        positions = [match.start() for pattern in patterns if (match := pattern.search(blob))]
        if positions:
            hits.append((min(positions), key))
    hits.sort()
    ordered = [GLOSSARY[key] for _, key in hits]
    return tuple(ordered[:limit] if limit is not None else ordered)


def concept_for_term(term: str) -> str | None:
    """Point a one-line definition at the concept note that teaches it properly.

    A glossary entry is deliberately short — one or two sentences. When the idea
    behind it has a full note, this returns that note's key so the reader can go
    deeper instead of stopping at the definition.

    Parameters
    ----------
    term:
        The term or any of its recorded aliases.

    Returns
    -------
    str or None
        A key into :data:`~buildml.explain.CONCEPT_NOTES`, or ``None`` when the
        term is unknown or has no curated concept.

    Notes
    -----
    This consults the curated ``CONCEPT_FOR_TERM`` table only.
    :func:`buildml.explain.learn` falls back to the vocabulary each concept note
    declares, so it resolves considerably more terms than this function does.
    """
    entry = lookup(term)
    if entry is None:
        return None
    for candidate in (entry.term, *entry.also_called):
        key = CONCEPT_FOR_TERM.get(candidate)
        if key:
            return key
    return None


def all_terms() -> tuple[GlossaryTerm, ...]:
    """List the whole glossary, for browsing rather than looking one term up.

    Useful for rendering a reference page or checking coverage. For teaching a
    single term, prefer :func:`buildml.explain.learn`, which returns the
    definition together with the concept that explains it and what to read
    first.

    Returns
    -------
    tuple of ~buildml.explain.schemas.GlossaryTerm
        Every entry, ordered alphabetically by term.
    """
    return tuple(GLOSSARY[key] for key in sorted(GLOSSARY))


__all__ = [
    "CONCEPT_FOR_TERM",
    "GLOSSARY",
    "all_terms",
    "concept_for_term",
    "detect_terms",
    "lookup",
    "require",
]
