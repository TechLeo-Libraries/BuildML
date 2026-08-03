# ruff: noqa: E501
"""Beginner layers for natural-language processing."""

from __future__ import annotations

from buildml.explain.beginner._builder import CORE, FOUNDATION, BeginnerLayer, _index, _layer

NLP_BEGINNER: dict[str, BeginnerLayer] = _index(
    _layer(
        "nlp-text-normalization",
        plain=(
            "Computers do not see 'Great!!!' and 'great' as the same word. Normalization is the tidying "
            "pass that makes them match: strip out HTML and web addresses, lower the case, remove "
            "punctuation, squash repeated characters. Tokenization then chops the tidied text into the "
            "individual words the model will actually count."
        ),
        analogy=(
            "Washing and chopping vegetables before cooking. It is not the cooking, but every recipe "
            "assumes it happened, and it has to happen the same way every time."
        ),
        steps=(
            "Character cleanup runs first, in a fixed order you choose: HTML, URLs, emails, case, accents, numbers, punctuation, repeats, whitespace.",
            "Tokenization splits the cleaned text into words.",
            "Optional extras: drop stopwords ('the', 'and'), filter very short tokens, reduce words to stems or dictionary forms.",
            "BuildML records every choice in a normalization plan.",
            "That plan travels inside the saved model, so scoring preprocesses text exactly as fitting did.",
        ),
        use=(
            "Always, before any text modelling. There is no 'raw text' path.",
            "Adjust the steps to your text: strip HTML for scraped pages, keep numbers for product data.",
        ),
        avoid=(
            "Do not strip punctuation before sentiment scoring; exclamation marks and capitals carry emphasis the scorer uses.",
            "Do not remove stopwords blindly: 'not' and 'no' are stopwords in many lists and completely reverse meaning.",
        ),
        myths=(
            (
                "Normalization must happen after the split to avoid leakage.",
                "It learns nothing from your data: it is fixed text substitution. What must be train-only is anything counted, like the vocabulary or the document frequencies.",
            ),
            (
                "More aggressive cleaning is better.",
                "Every step deletes information. Stripping numbers removes dosages and model numbers; stripping accents merges words that differ in meaning.",
            ),
        ),
        example=(
            "session.fit_text_classifier(",
            "    text_column='review',",
            "    normalize_steps=['strip_html', 'lowercase', 'collapse_whitespace'],",
            "    stopword_language='en',",
            ")",
            "print(session.nlp_text_plan.normalize_plan.to_dict())",
        ),
        check=(
            "Which of your normalization steps could delete something meaningful?",
            "Will the exact same steps run at scoring time? (Yes, if the plan travels with the model.)",
        ),
        tools=("fit_text_classifier", "fit_topics", "extract_keyphrases", "summarize_text"),
        terms=("tokenization", "stopword", "stemming", "corpus"),
        difficulty=FOUNDATION,
    ),
    _layer(
        "nlp-document-representation",
        plain=(
            "A model cannot read. Text has to become a row of numbers first. There are two broad ways: "
            "count which words appear and how unusual they are (bag-of-words / TF-IDF), or hand the text to "
            "a pretrained neural network that outputs a vector capturing meaning (an embedding)."
        ),
        analogy=(
            "Describing a book by a tally of which words it uses, versus by a summary of what it is about. "
            "The tally is transparent and literal; the summary is richer and impossible to trace back."
        ),
        steps=(
            "TF-IDF counts each word in a document, then downweights words that appear in many documents.",
            "The vocabulary and the document frequencies are learned from training documents only.",
            "N-grams let you count phrases too: `ngram_range=(1, 2)` counts single words and adjacent pairs.",
            "Hashing skips the vocabulary entirely, which saves memory and makes features unnameable.",
            "Embeddings run each document through a frozen pretrained model and take the output vector.",
        ),
        use=(
            "TF-IDF when you want speed, interpretability, and a strong baseline: which is most of the time.",
            "Character n-grams when your text has typos, mixed morphology, or no clear word boundaries.",
            "Embeddings when paraphrase matters and different words mean the same thing.",
        ),
        avoid=(
            "Do not fit the vectorizer before splitting; the vocabulary and document frequencies would be learned from your test set.",
            "Do not choose embeddings and then ask which words the model looked at: those positions have no word attached.",
        ),
        myths=(
            (
                "Embeddings always beat TF-IDF.",
                "On focused classification tasks with enough labelled data, a TF-IDF linear model frequently wins, trains in seconds, and can be explained word by word.",
            ),
            (
                "Setting min_df=1 keeps more signal.",
                "It keeps every typo and one-off token. Those cannot generalize by definition, and they bloat the feature space.",
            ),
        ),
        example=(
            "session.fit_text_classifier(vectorizer='tfidf', ngram_range=(1, 2), min_df=2)",
            "report = session.evaluate_text_classifier(partition='test')",
            "print(report.metrics, report.oov_rate)",
        ),
        check=(
            "What is your out-of-vocabulary rate on holdout? A high number means the model literally cannot see much of the test text.",
            "Do you need to explain individual predictions? If so, avoid hashing and embeddings.",
        ),
        tools=("fit_text_classifier", "evaluate_text_classifier", "interpret_text_prediction"),
        terms=("TF-IDF", "n-gram", "embedding", "vocabulary", "out-of-vocabulary"),
        difficulty=CORE,
    ),
    _layer(
        "nlp-token-attribution",
        plain=(
            "For a linear text model you can say exactly which words pushed a prediction which way: not "
            "an estimate, but arithmetic. The model's score is a sum of one small contribution per word, "
            "and attribution simply sorts those contributions."
        ),
        analogy=(
            "An itemized receipt. The total is not a mystery; it is the lines added up, and you can point "
            "at the expensive one."
        ),
        steps=(
            "Each word in the document has a feature value and a model weight.",
            "Multiply them and you get that word's contribution to the score.",
            "Add all contributions plus the bias and you recover the score exactly.",
            "Positive contributions push toward the class; negative ones push away.",
            "BuildML also reports global top words per class: the highest weights regardless of any document.",
        ),
        use=(
            "When someone asks why a specific document was classified the way it was.",
            "For debugging: attribution quickly reveals that your model is keying on a boilerplate footer.",
        ),
        avoid=(
            "Do not expect attribution after using hashing, embeddings, or a transformer encoder: BuildML refuses, because those features have no word names.",
            "Do not read the global top-word list as an explanation of one prediction; a high-weight word contributes nothing if it does not appear.",
        ),
        myths=(
            (
                "A high coefficient means the word causes the outcome.",
                "It means the word is statistically associated with the label in your training data. 'Refund' predicting complaints does not cause complaints.",
            ),
            (
                "Attribution is an approximation like SHAP.",
                "For a linear head it is exact: the contributions provably sum to the score. That is why BuildML refuses rather than approximating when the representation does not allow it.",
            ),
        ),
        example=(
            "session.fit_text_classifier(estimator='logistic', vectorizer='tfidf')",
            "expl = session.interpret_text_prediction(top_k=12)",
            "print(expl.document_attributions[0])",
            "print(expl.global_top_tokens)",
        ),
        check=(
            "Do the top contributing words make sense, or are they boilerplate?",
            "Is your representation invertible? Hashing and embeddings are not.",
        ),
        tools=("interpret_text_prediction", "fit_text_classifier", "feature_importance"),
        terms=("attribution", "coefficient", "TF-IDF", "vocabulary"),
        difficulty=CORE,
    ),
    _layer(
        "nlp-topic-models",
        plain=(
            "A topic model reads a pile of documents with no labels and finds recurring vocabulary "
            "patterns. Each 'topic' it returns is a ranked list of words that tend to appear together. It "
            "does not name the topic: that is your job."
        ),
        analogy=(
            "Sorting a huge unlabelled pile of correspondence into stacks that feel similar. The machine "
            "makes the stacks; you read them and decide what each one is about."
        ),
        steps=(
            "Documents are turned into word counts.",
            "The model factors that matrix into topics: each document is a mixture of topics, each topic is a weighted word list.",
            "NMF minimizes reconstruction error; LDA fits a probabilistic mixture. Both are fitted on training documents only.",
            "BuildML labels each topic with its top words purely so the output is readable.",
            "Coherence scores each topic by whether its top words genuinely co-occur.",
        ),
        use=(
            "For exploring a corpus you do not yet understand.",
            "To produce topic-weight features for a downstream model, fitted on train and applied to holdout.",
        ),
        avoid=(
            "Do not treat topics as validated categories; the auto-generated label is a convenience, not a finding.",
            "Do not fit topics on the whole corpus and then use topic weights as model features: that is leakage through the back door.",
        ),
        myths=(
            (
                "The number of topics is discovered by the model.",
                "You choose it. Set it too high and one real theme fragments across several components; too low and distinct themes merge.",
            ),
            (
                "Low LDA perplexity means good topics.",
                "Perplexity measures fit to word counts and correlates poorly with whether topics make sense to a human. Coherence is the more useful signal.",
            ),
        ),
        example=(
            "session.fit_topics(method='nmf', n_topics=8, min_df=2, random_state=0)",
            "for topic in session.nlp_topic_plan.topics:",
            "    print(topic.terms, topic.coherence, topic.train_mass)",
            "session.assign_topics(partition='test')",
        ),
        check=(
            "Which topics have low coherence? Those are noise, not themes.",
            "Is boilerplate: signatures, disclaimers: dominating every topic?",
        ),
        tools=("fit_topics", "assign_topics", "extract_keyphrases"),
        terms=("topic model", "corpus", "coherence", "clustering"),
        difficulty=CORE,
    ),
    _layer(
        "nlp-keyphrases-vs-topics",
        plain=(
            "Keyphrase extraction pulls the standout phrases out of individual documents. Topic modelling "
            "finds patterns spanning the whole collection. Both are unsupervised descriptions, and they "
            "answer different questions: what is *this document* about, versus what is *this corpus* about."
        ),
        analogy=(
            "Highlighting the key line in each letter, versus sorting all the letters into themed folders. "
            "Related activities, different outputs."
        ),
        steps=(
            "TF-IDF keyphrases rank phrases that are frequent here and rare elsewhere.",
            "RAKE splits text at stopwords and scores the remaining chunks by how connected their words are.",
            "TextRank builds a graph of nearby words and finds the most central ones, like PageRank for text.",
            "Candidates exclude bare numbers and punctuation.",
            "Nothing is fitted or saved; each call is a fresh description.",
        ),
        use=(
            "As the cheapest possible summary of what a body of text contains, with no labels needed.",
            "As a sanity check alongside topic modelling: if they disagree wildly, something is off.",
        ),
        avoid=(
            "Do not claim precision or recall; without an annotated reference set those numbers cannot exist.",
            "Do not use holdout keyphrases to choose features or settings; describing holdout is fine, deciding from it is not.",
        ),
        myths=(
            (
                "Scores are comparable across methods.",
                "They are comparable only within a single call. A RAKE score of 8 and a TextRank score of 8 mean nothing to each other.",
            ),
            (
                "The top phrase is the most important thing in the document.",
                "It is the highest-scoring under one heuristic. Boilerplate frequently tops the ranking, which is itself informative.",
            ),
        ),
        example=(
            "result = session.extract_keyphrases(partition='train', method='rake', top_n=20)",
            "print(result.corpus_keyphrases)",
            "print(result.document_keyphrases[0])",
        ),
        check=(
            "Are your top phrases actual content, or headers and disclaimers?",
            "Do you want per-document description or corpus-level structure?",
        ),
        tools=("extract_keyphrases", "fit_topics", "summarize_text"),
        terms=("keyphrase", "corpus", "TF-IDF", "stopword"),
        difficulty=FOUNDATION,
    ),
    _layer(
        "nlp-lexicon-sentiment",
        plain=(
            "Lexicon sentiment scores text using a dictionary of words with positive or negative values, "
            "plus rules for the things that flip meaning: 'not good' is negative, 'VERY BAD!!' is worse than "
            "'bad', and everything after 'but' counts for more. It needs no training data at all."
        ),
        analogy=(
            "Marking an essay with a fixed rubric rather than by judgement. Transparent, instantly "
            "available, and blind to anything the rubric does not mention."
        ),
        steps=(
            "Look up each word's valence in the lexicon and add them up.",
            "A negator nearby flips and dampens the following word.",
            "An intensifier scales it, with the effect fading over distance.",
            "Exclamation marks and capitals add emphasis; question marks dampen.",
            "The total is squashed into a bounded compound score and thresholded into positive, negative, or neutral.",
        ),
        use=(
            "On day one, before you have any labels. It is the correct baseline for any sentiment work.",
            "When you need to explain a score word by word.",
        ),
        avoid=(
            "Do not use it on non-English text without checking; the shipped lexicon covers English only.",
            "Do not use it for sarcasm, irony, or domain jargon where ordinary words carry unusual weight.",
        ),
        myths=(
            (
                "A large neutral share means the text is balanced.",
                "It often means the lexicon recognized nothing. That is why BuildML reports the matched-term rate: near zero means ignorance, not balance.",
            ),
            (
                "The compound score is a probability.",
                "It is a bounded ordinal score between -1 and 1. It orders documents; it does not state a likelihood.",
            ),
        ),
        example=(
            "base = session.analyze_sentiment(backend='lexicon')",
            "print(base.matched_term_rate, base.label_counts)",
            "session.fit_text_classifier(text_column='review')",
            "tuned = session.analyze_sentiment(backend='supervised')",
        ),
        check=(
            "What is your matched-term rate? Below about half, the scores are mostly guesswork.",
            "Have you confirmed your corpus is English?",
        ),
        tools=("analyze_sentiment", "detect_language", "fit_text_classifier"),
        terms=("sentiment", "lexicon", "baseline", "corpus"),
        difficulty=FOUNDATION,
    ),
    _layer(
        "nlp-rule-vs-statistical-ner",
        plain=(
            "Entity extraction finds the dates, amounts, emails, and names inside text. You can do it with "
            "patterns you write down, which find exactly what you described and nothing else, or with a "
            "trained model, which finds more but also invents things."
        ),
        analogy=(
            "A checklist inspector versus an experienced one. The checklist never flags something outside "
            "it and never flags it wrongly. The experienced inspector notices more, and is occasionally "
            "wrong."
        ),
        steps=(
            "The rules backend applies precision-first patterns for dates, money, percentages, emails, URLs, phone numbers, and identifiers.",
            "You can add gazetteers: your own lists of product names or codes, matched as whole words.",
            "Overlapping matches are resolved deterministically by length and pattern priority.",
            "Every span reports which rule found it, so a false positive is traceable and fixable.",
            "The spaCy backend runs a trained model instead, requiring the industry extra and a downloaded model.",
        ),
        use=(
            "Rules when a false positive is expensive and the entity types have recognizable shapes.",
            "spaCy when you need people and organizations, which no pattern can reliably capture.",
        ),
        avoid=(
            "Do not add a gazetteer term that is a common word; you will flood the output.",
            "Do not read a zero count for a type as 'the corpus contains none': it may mean no rule covers that type.",
        ),
        myths=(
            (
                "Rules are the fallback and the model is the real solution.",
                "For extraction feeding an automated decision, precision usually matters more than coverage. Rules are frequently the correct engineering answer.",
            ),
            (
                "spaCy labels mean the same as rule labels.",
                "BuildML normalizes them to a common set, but the underlying definitions differ. Check the mapping before comparing counts.",
            ),
        ),
        example=(
            "found = session.extract_entities(",
            "    backend='rules',",
            "    gazetteers={'PRODUCT': ['widget-9', 'widget-12']},",
            ")",
            "print(found.label_counts)",
            "print(found.spans[0].source, found.spans[0].start, found.spans[0].end)",
        ),
        check=(
            "For each entity type you need: is there a pattern covering it?",
            "What is the cost of a false positive in your workflow?",
        ),
        tools=("extract_entities", "analyze_sentiment", "fit_text_classifier"),
        terms=("named entity recognition", "gazetteer", "precision", "recall"),
        difficulty=CORE,
    ),
    _layer(
        "nlp-extractive-summarization",
        plain=(
            "Extractive summarization picks the most representative sentences out of a document and hands "
            "them back unchanged, in their original order. It never writes anything, which means it can "
            "never state something the document does not say."
        ),
        analogy=(
            "Highlighting three sentences in an article versus writing your own précis. The highlighter "
            "cannot misquote."
        ),
        steps=(
            "Split the document into sentences, carefully: 'Dr. Smith' is not two sentences.",
            "Measure how similar each sentence is to every other one.",
            "Run a centrality algorithm: the most representative sentence is the one most like the rest.",
            "Take the top `n_sentences` and emit them in document order.",
            "Compare against the lead baseline, which just takes the first few sentences.",
        ),
        use=(
            "When factual safety matters and an invented sentence would be unacceptable.",
            "For long documents where a reader needs a quick way in.",
        ),
        avoid=(
            "Do not use it on lists, tables, or single-sentence documents; sentence boundaries carry no meaning there.",
            "Do not report ROUGE scores without reference summaries: there is nothing to compare against.",
        ),
        myths=(
            (
                "The graph methods obviously beat taking the first few sentences.",
                "For news-like text the lead baseline is famously hard to beat. BuildML ships it precisely so you can check rather than assume.",
            ),
            (
                "This is text generation.",
                "Nothing is generated. Every word in the output appears verbatim in the input, which is the whole safety argument for the approach.",
            ),
        ),
        example=(
            "lead = session.summarize_text(method='lead', n_sentences=3)",
            "rank = session.summarize_text(method='textrank', n_sentences=3)",
            "print(lead.mean_compression, rank.mean_compression)",
            "print(rank.selected_sentence_indices[0])",
        ),
        check=(
            "Does the graph method actually beat lead on your documents, judged by reading them?",
            "Are your documents long enough for selection to mean anything?",
        ),
        tools=("summarize_text", "extract_keyphrases", "fit_topics"),
        terms=("summarization", "extractive", "baseline", "corpus"),
        difficulty=FOUNDATION,
    ),
    _layer(
        "nlp-language-identification",
        plain=(
            "Language identification works out what language each document is written in. It first looks "
            "at the alphabet, then at the little function words. When there is too little text to be sure, "
            "it answers 'undetermined' rather than guessing."
        ),
        analogy=(
            "Recognizing a language from a glance at the page: the script narrows it down enormously, and "
            "a handful of common short words usually settles it."
        ),
        steps=(
            "Measure which Unicode scripts the characters belong to: Latin, Cyrillic, Han, Arabic, and so on.",
            "For Latin script, score discriminative function words such as 'the', 'le', 'der', 'el'.",
            "Below a minimum length or a minimum score, return 'und' for undetermined.",
            "Report per-document codes, the dominant language, and the undetermined rate.",
            "The langdetect backend delegates to a trained n-gram model instead.",
        ),
        use=(
            "Before choosing stopword lists, stemmers, or a sentiment lexicon: all of them are language-specific.",
            "As a cheap data-quality screen on any new text corpus.",
        ),
        avoid=(
            "Do not force a label on very short documents; 'und' is the honest answer and forcing it produces confident nonsense.",
            "Do not expect one label to describe a code-switched document.",
        ),
        myths=(
            (
                "'und' means an unknown or exotic language.",
                "It means there was not enough evidence to decide. A high undetermined rate almost always means short documents.",
            ),
            (
                "The confidence value is a probability.",
                "It is a relative marker score. Use it to rank certainty, not to reason about likelihoods.",
            ),
        ),
        example=(
            "langs = session.detect_language(partition='all')",
            "print(langs.dominant_language, langs.undetermined_rate, langs.language_counts)",
            "if langs.dominant_language != 'en':",
            "    ...  # reconsider stopword_language and the sentiment backend",
        ),
        check=(
            "Is your corpus really single-language?",
            "Are URLs, code, or identifiers dominating the tokens and confusing detection?",
        ),
        tools=("detect_language", "profile_text_corpus", "analyze_sentiment"),
        terms=("language identification", "corpus", "tokenization", "stopword"),
        difficulty=FOUNDATION,
    ),
    _layer(
        "nlp-corpus-contamination",
        plain=(
            "Text collections are full of copies: reposts, syndicated articles, templated emails, the same "
            "review submitted twice. If a document in your test set also appears in training, the model "
            "does not have to generalize to score it. Random splitting cannot see this, because it only "
            "looks at rows."
        ),
        analogy=(
            "An exam where some questions were on the practice paper. The scores look excellent and mean "
            "nothing about whether the material was learned."
        ),
        steps=(
            "BuildML fingerprints each document after normalization and finds exact matches across the split.",
            "It also finds near-duplicates: pairs whose character-level similarity exceeds a threshold.",
            "It reports the holdout out-of-vocabulary rate as the complementary signal.",
            "Findings surface as warnings on the result and in the walkthrough.",
            "Nothing is removed automatically: deduplicating stays your explicit decision.",
        ),
        use=(
            "Always, right after splitting a text corpus and before fitting anything.",
            "Especially when your documents come from the web, from email, or from any templated source.",
        ),
        avoid=(
            "Do not deduplicate silently and report the resulting clean score; the decision and its consequences belong in the record.",
            "Do not quote a near-duplicate count without the threshold that produced it: the number is meaningless alone.",
        ),
        myths=(
            (
                "A random split protects against this.",
                "A random split guarantees that duplicate documents land on both sides roughly in proportion. It is the cause, not the cure.",
            ),
            (
                "A very low out-of-vocabulary rate is good news.",
                "A very low holdout out-of-vocabulary rate alongside many near-duplicates is the classic contamination signature. Your holdout is too similar to training.",
            ),
        ),
        example=(
            "session.split(test_size=0.2, random_state=0)",
            "profile = session.profile_text_corpus(near_duplicate_threshold=0.9)",
            "print(profile.exact_overlap, profile.near_duplicate_pairs, profile.holdout_oov_rate)",
        ),
        check=(
            "How many holdout documents appear verbatim in training?",
            "If you removed the duplicates, would the holdout score drop?",
        ),
        tools=("profile_text_corpus", "split", "fit_text_classifier", "detect_language"),
        terms=("contamination", "near-duplicate", "holdout", "leakage", "out-of-vocabulary"),
        difficulty=CORE,
    ),
    _layer(
        "nlp-vs-rag",
        plain=(
            "Four BuildML surfaces touch text and they are not interchangeable. NLP models a text column on "
            "your dataset. RAG retrieves passages from a document corpus so an answer can cite them. "
            "`text_features` turns a text column into ordinary columns for a tabular model. The Torch text "
            "path fine-tunes a neural network."
        ),
        analogy=(
            "A microscope, a library catalogue, a blender, and a workshop. All of them process material; "
            "picking the wrong one does not slow you down, it produces the wrong thing."
        ),
        steps=(
            "Label on the row, text as evidence? Use `fit_text_classifier`.",
            "Answer lives in a document you must find and cite? Use the RAG surface.",
            "Text is one signal among many tabular columns? Use `text_features`.",
            "Need to update neural network weights on your own text? Use the Torch text path.",
            "Each has its own Session state, its own bundle format, and its own load-time validation.",
        ),
        use=(
            "NLP for document classification, topics, sentiment, entities, and summaries.",
            "RAG for grounded question answering over your documents.",
        ),
        avoid=(
            "Do not route document classification through RAG retrieval; there is no holdout classification metric there.",
            "Do not use `text_features` and then claim word-level explanations of the tabular model.",
        ),
        myths=(
            (
                "They all index text, so the artifacts should be interchangeable.",
                "Each bundle validates its own format tag on load. That refusal is what stops a RAG index from silently standing in for a fitted classifier.",
            ),
            (
                "The core NLP path needs the transformer extras.",
                "TF-IDF classification, topics, keyphrases, rule entities, lexicon sentiment, summarization, and language detection all work with no extra installs.",
            ),
        ),
        example=(
            "session.fit_text_classifier(text_column='ticket')   # classify documents",
            "session.rag_ingest_corpus(documents)                # grounded answers",
            "session.text_features(column='notes')               # tabular expansion",
        ),
        check=(
            "Is your question about a labelled row, or about finding a document?",
            "Which bundle type will your deployment need to load?",
        ),
        tools=("fit_text_classifier", "rag_ingest_corpus", "text_features", "save_nlp_bundle"),
        terms=("RAG", "corpus", "bundle", "text features"),
        difficulty=FOUNDATION,
    ),
    _layer(
        "nlp-bundle-boundary",
        plain=(
            "The fitted text model: vectorizer, classifier head, topic model, and crucially the "
            "normalization plan: saves as an NLP bundle. A Session checkpoint stores your data and "
            "workflow and contains none of it."
        ),
        analogy=(
            "The trained reader and the pile of documents are separate things. Boxing up the documents "
            "does not preserve the reader's training."
        ),
        steps=(
            "Fit a text classifier, a topic model, or both.",
            "Call `save_nlp_bundle(path)`: the normalization plan travels inside it.",
            "Reload with `load_nlp_bundle(path)` in a Session exposing the same text column name.",
            "Loading clears stale fit, evaluation, prediction, and interpretation results so nothing is misattributed.",
            "Evaluate on holdout after loading rather than trusting the recorded metrics for a new dataset.",
        ),
        use=(
            "When text scoring runs in a service separate from where the model was fitted.",
            "When you need to guarantee the exact same preprocessing months later.",
        ),
        avoid=(
            "Do not ship a fitted vectorizer without its normalization plan; the preprocessing mismatch will quietly destroy accuracy.",
            "Do not expect a RAG bundle to load as an NLP bundle: the formats are validated on load.",
        ),
        myths=(
            (
                "A preprocessing mismatch would raise an error.",
                "It will not. Every metric still computes, and the numbers are just worse. That silent failure is exactly why the plan lives inside the bundle.",
            ),
            (
                "The checkpoint would include the NLP model since it is Session state.",
                "Checkpoints carry data, roles, splits, and history. Learner state lives in bundles so each artifact can be validated independently.",
            ),
        ),
        example=(
            "session.fit_text_classifier(text_column='review')",
            "session.save_nlp_bundle('artifacts/review-clf')",
            "svc = Session.ingest(incoming).load_nlp_bundle('artifacts/review-clf')",
            "svc.evaluate_text_classifier(partition='test')",
        ),
        check=(
            "Does the loading Session use the same text column name?",
            "Is the bundle directory complete, with metadata and payloads?",
        ),
        tools=("save_nlp_bundle", "load_nlp_bundle", "fit_text_classifier", "checkpoint_save"),
        terms=("bundle", "checkpoint", "corpus", "TF-IDF"),
        difficulty=CORE,
    ),
)

__all__ = ["NLP_BEGINNER"]
