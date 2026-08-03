# ruff: noqa: E501
"""Natural-language processing concept notes."""

from __future__ import annotations

from buildml.explain.concepts._builder import _note
from buildml.explain.schemas import ConceptNote

NLP_NOTES: dict[str, ConceptNote] = {
    note.key: note
    for note in (
        _note(
            key="nlp-text-normalization",
            title="Text normalization and tokenization (fit-free)",
            summary="Character cleanup plus tokenization learns nothing from the corpus, so it is safe before a split: but it must be identical at fit and score time.",
            definition=(
                "Normalization is an ordered pipeline of character-level steps "
                "(strip_html, strip_urls, strip_emails, lowercase, "
                "strip_accents, strip_numbers, strip_punctuation, "
                "collapse_repeats, collapse_whitespace) followed by "
                "tokenization into words, with optional stopword removal, "
                "length filters, stemming, and lemmatization. BuildML resolves "
                "the choices once into a TextNormalizePlan and embeds that plan "
                "in every NLP artifact, so scoring reproduces the exact "
                "preprocessing the model was fitted with."
            ),
            intuition=(
                "Two documents that mean the same thing should look the same to "
                "the model. Normalization is the set of edits that make "
                "'Great!!!' and 'great' the same token, and tokenization is the "
                "decision about what counts as one unit of text."
            ),
            formal_idea=(
                "T: string -> string is a fixed composition of regular-expression "
                "substitutions; K: string -> list[str] tokenizes. Neither depends "
                "on corpus statistics, so T and K commute with the train/test "
                "split. Anything estimated from counts (vocabulary, document "
                "frequency, idf) is separate and train-only."
            ),
            why_it_matters=(
                "A preprocessing mismatch between fit and score silently destroys accuracy while every metric still 'works'.",
                "Because normalization is fit-free, it is not a leakage risk: which makes it important to say clearly, so the real risk (vocabulary fitting) is not confused with it.",
                "Stemming and lemmatization change what a token means; the plan records which backend produced them.",
            ),
            how_buildml_uses=(
                "build_normalize_plan resolves config into a serializable TextNormalizePlan.",
                "The same plan drives fit_text_classifier, fit_topics, extract_keyphrases, and summarize_text.",
                "The plan travels inside buildml.nlp_bundle.v1 so a reloaded model preprocesses identically.",
            ),
            interpretation_rules=(
                "Read plan.normalize_plan.to_dict() to see exactly which steps ran.",
                "n_stopwords tells you how much vocabulary was discarded before counting.",
                "stem_backend='native-suffix' means NLTK was absent and conservative built-in rules ran instead.",
            ),
            assumptions=(
                "Documents are text-like; numeric columns are refused rather than coerced.",
                "One text column at a time is the unit of work.",
            ),
            failure_modes=(
                "Aggressive stopword removal deleting the signal (negations, 'not', 'no').",
                "strip_numbers removing meaningful identifiers such as model numbers or dosages.",
                "Lemmatization requested without the NLTK WordNet corpus downloaded.",
            ),
            anti_patterns=(
                "Normalizing train and holdout with different settings.",
                "Calling normalization 'leakage' and skipping it before the split, then applying different settings later.",
                "Stripping punctuation before sentiment scoring and then wondering why emphasis disappeared.",
            ),
            worked_example_pattern=(
                "fit_text_classifier(normalize_steps=['strip_html', 'lowercase', 'collapse_whitespace'], stopword_language='en') -> inspect nlp_text_plan.normalize_plan.",
            ),
            related_concepts=(
                "nlp-document-representation",
                "nlp-corpus-contamination",
                "leakage-boundary",
            ),
        ),
        _note(
            key="nlp-document-representation",
            title="Document representations: bag-of-n-grams, embeddings, encoders",
            summary="Every text model turns documents into vectors; the choice decides whether the model is interpretable, how it generalizes, and what it costs.",
            definition=(
                "A document representation maps text to a numeric vector. "
                "BuildML offers three: train-fitted bag-of-n-grams (TF-IDF, "
                "counts, or hashing over word or character n-grams), frozen "
                "sentence-transformer embeddings, and a frozen mean-pooled "
                "Hugging Face encoder. Only the first has an invertible "
                "vocabulary, meaning a feature position can be named as a token."
            ),
            intuition=(
                "Bag-of-n-grams asks 'which words appear, and how surprising are "
                "they?': sparse, fast, and readable. An embedding asks 'what "
                "does this document mean?': dense, better on paraphrase, and "
                "unreadable position by position."
            ),
            formal_idea=(
                "TF-IDF: x_j = tf(t_j, d) * log((1 + N) / (1 + df(t_j))) + 1, "
                "then L2-normalized, with tf and df estimated on train alone. "
                "Hashing: x_h(t) with h a fixed hash, so no vocabulary is stored "
                "and collisions are possible. Embedding: x = pool(f_theta(d)) for "
                "frozen theta."
            ),
            why_it_matters=(
                "Interpretability is a property of the representation, not of the classifier: nothing can name a hashed or pooled feature.",
                "Out-of-vocabulary rate on holdout tells you how much of the text the fitted vocabulary simply cannot see.",
                "Character n-grams survive typos and morphology where word n-grams fail.",
            ),
            how_buildml_uses=(
                "fit_text_classifier(backend=..., vectorizer=..., analyzer=..., ngram_range=...).",
                "Vocabulary, document frequencies, and idf are fitted on Session train only.",
                "The fitted vectorizer is stored on NlpTextPlan and reused verbatim for predict and evaluate.",
            ),
            interpretation_rules=(
                "n_features is the representation width; vocabulary_size is 0 for hashing because nothing is stored.",
                "oov_rate on eval/predict is the share of holdout tokens absent from the train vocabulary.",
                "sublinear_tf=True damps repeated terms, so a word said ten times is not ten times the evidence.",
            ),
            assumptions=(
                "Documents in holdout resemble train documents in language and register.",
                "min_df / max_df are chosen with the corpus size in mind.",
            ),
            failure_modes=(
                "max_features far below the useful vocabulary, silently truncating signal.",
                "min_df=1 on a small corpus, producing a vocabulary of one-off tokens that cannot generalize.",
                "Hash collisions conflating unrelated tokens in the hashing vectorizer.",
                "Embedding or encoder backend requested without buildml[nlp] installed.",
            ),
            anti_patterns=(
                "Fitting the vectorizer on the whole frame before splitting.",
                "Choosing embeddings for a task where a TF-IDF linear model is both stronger and readable.",
                "Reporting 'the model looks at the word X' after fitting a hashing vectorizer.",
            ),
            worked_example_pattern=(
                "fit_text_classifier(vectorizer='tfidf', ngram_range=(1, 2)) -> evaluate_text_classifier() -> read metrics and oov_rate.",
            ),
            related_concepts=(
                "nlp-text-normalization",
                "nlp-token-attribution",
                "nlp-corpus-contamination",
                "nlp-vs-rag",
                "leakage-boundary",
            ),
        ),
        _note(
            key="nlp-token-attribution",
            title="Token attribution for linear document models",
            summary="For a linear head the decision score decomposes exactly into per-token terms, so attribution is arithmetic: not an approximation, and not always available.",
            definition=(
                "For a linear classifier the score of class c on document d is "
                "bias_c + sum over features j of coef[c, j] * x_j(d). Each term "
                "coef[c, j] * x_j(d) is that feature's exact contribution. When "
                "the feature index j can be mapped back to a token (TF-IDF or "
                "count vectorizers), those contributions are readable as words "
                "and phrases."
            ),
            intuition=(
                "The model's answer is a sum of small votes, one per token that "
                "actually appeared. Attribution just sorts the votes."
            ),
            formal_idea=(
                "s_c(d) = b_c + <w_c, x(d)>; contribution_j = w_{c,j} * x_j(d), "
                "and sum_j contribution_j + b_c = s_c(d) exactly. Global weights "
                "w_c rank vocabulary independent of occurrence."
            ),
            why_it_matters=(
                "Exact decomposition means you can audit a decision rather than narrate a plausible story about it.",
                "Refusing attribution for hashing, embedding, and encoder representations is the honest answer, because those positions have no token name.",
                "Distinguishing per-document contributions from global weights prevents the common error of reading a global list as an explanation of one prediction.",
            ),
            how_buildml_uses=(
                "interpret_text_prediction returns per-document TokenAttribution rows plus per-class global top tokens.",
                "Naive-Bayes heads use centred log-likelihoods and are labelled as ranking evidence, not additive terms.",
                "Binary linear heads are expanded to two signed rows so the requested class is always explained in its own orientation.",
            ),
            interpretation_rules=(
                "A positive contribution pushes the document toward the target class; a negative one pushes it away.",
                "value is the feature value in the document, weight is the model coefficient; a large weight with value 0 contributes nothing.",
                "Global top tokens ignore frequency, so a high-weight token may be rare and practically irrelevant.",
            ),
            assumptions=(
                "The head is linear (logistic, linear SVM, SGD) or naive Bayes.",
                "The plan carries feature names, which requires fitting through fit_text_classifier.",
            ),
            failure_modes=(
                "Requesting attribution after fitting with vectorizer='hashing' (refused).",
                "Requesting attribution on an embedding or transformer backend (refused).",
                "Reading LinearSVC margins as probabilities.",
            ),
            anti_patterns=(
                "Approximating attributions for a non-invertible representation to avoid returning an error.",
                "Presenting global coefficient rankings as per-document explanations.",
                "Treating high coefficients as causal effects of words.",
            ),
            worked_example_pattern=(
                "fit_text_classifier(estimator='logistic') -> interpret_text_prediction(top_k=12) -> compare document_attributions with global_top_tokens.",
            ),
            related_concepts=(
                "nlp-document-representation",
                "nlp-text-normalization",
                "feature-importance",
            ),
        ),
        _note(
            key="nlp-topic-models",
            title="Topic models are ranked term lists, not named categories",
            summary="NMF and LDA decompose a document-term matrix into components; the components are term rankings you must interpret, and coherence is the honest quality signal.",
            definition=(
                "A topic model factors the document-term matrix into a "
                "document-topic matrix and a topic-term matrix. NMF minimizes "
                "reconstruction error on TF-IDF with non-negativity; LDA fits a "
                "probabilistic mixture over counts. A 'topic' is a weighted list "
                "of terms; BuildML labels each one with its top terms purely for "
                "readability."
            ),
            intuition=(
                "Documents are mixtures of recurring vocabulary patterns. The "
                "model finds those patterns; naming them is your job, not the "
                "model's."
            ),
            formal_idea=(
                "NMF: min over W >= 0, H >= 0 of ||X - W H||_F. LDA: "
                "p(w | d) = sum_k p(w | k) p(k | d) with Dirichlet priors. "
                "NPMI coherence for a topic's top terms: mean over term pairs of "
                "log(p(i, j) / (p(i) p(j))) / -log p(i, j), which is near 1 for "
                "terms that genuinely co-occur."
            ),
            why_it_matters=(
                "Unsupervised structure invites over-claiming; coherence gives a number to argue with.",
                "Fitting the vectorizer and decomposition on train only makes assign_topics on holdout a pure transform, which keeps topic features usable downstream.",
                "n_topics is a modelling decision, not a discovered truth.",
            ),
            how_buildml_uses=(
                "fit_topics(method='nmf' | 'lda') fits on Session train and reports per-topic NPMI coherence and train mass.",
                "assign_topics transforms a partition into per-document topic weights plus dominant-topic shares.",
                "The fitted topic plan is persisted in buildml.nlp_bundle.v1 alongside any text classifier.",
            ),
            interpretation_rules=(
                "Read terms and weights, not the auto-generated label, when deciding what a topic is.",
                "Low or negative coherence means the top terms rarely co-occur; treat the topic as noise.",
                "train_mass shows how much of the train corpus loads on the topic; tiny topics are usually artifacts.",
                "LDA perplexity is comparable only across runs on the same vectorized corpus.",
            ),
            assumptions=(
                "Enough documents to estimate term co-occurrence (BuildML requires a minimum and warns near it).",
                "min_df / max_df have removed both one-off tokens and near-universal boilerplate.",
            ),
            failure_modes=(
                "n_topics far above the number of real themes, splitting one theme across many components.",
                "Boilerplate (signatures, disclaimers) dominating every topic.",
                "Reading LDA perplexity as a semantic quality score.",
            ),
            anti_patterns=(
                "Fitting topics on the full corpus and then using topic features as model inputs.",
                "Presenting the auto-generated top-term label as a validated category name.",
                "Comparing coherence across different vectorizer settings as if it were an absolute scale.",
            ),
            worked_example_pattern=(
                "fit_topics(method='nmf', n_topics=8, min_df=2) -> inspect coherence -> assign_topics(partition='test').",
            ),
            related_concepts=(
                "nlp-document-representation",
                "nlp-keyphrases-vs-topics",
                "cluster-validity-not-truth",
                "unsupervised-train-fit-holdout-assign",
            ),
        ),
        _note(
            key="nlp-keyphrases-vs-topics",
            title="Keyphrases describe documents; topics describe corpora",
            summary="TF-IDF, RAKE, and TextRank rank phrases without supervision: useful description, but no precision or recall can be claimed without a gold set.",
            definition=(
                "Keyphrase extraction scores candidate phrases inside documents. "
                "TF-IDF ranks corpus-weighted n-grams; RAKE scores "
                "stopword-delimited candidates by degree over frequency; "
                "TextRank runs PageRank over a sliding-window word graph and "
                "sums word centrality across a phrase. All three are "
                "unsupervised and stateless."
            ),
            intuition=(
                "A keyphrase is a phrase that is common in this document and "
                "unusual elsewhere, or that sits at the centre of the document's "
                "word graph. A topic is a pattern shared across many documents."
            ),
            formal_idea=(
                "RAKE: score(word) = (degree(word) + freq(word)) / freq(word), "
                "and score(phrase) = sum of member word scores. TextRank: "
                "s(v) = (1 - d) / |V| + d * sum over neighbours u of "
                "s(u) * w(u, v) / sum_z w(u, z)."
            ),
            why_it_matters=(
                "Keyphrases are the cheapest honest summary of what a corpus is about, and they need no labels.",
                "Because there is no ground truth, reporting a quality metric would be fabrication; only within-run comparability is claimed.",
                "Running extraction on holdout text is description, not model selection: but reading holdout still informs the analyst, so it is disclosed.",
            ),
            how_buildml_uses=(
                "extract_keyphrases(method='tfidf' | 'rake' | 'textrank') returns corpus-level and per-document rankings.",
                "Candidates exclude bare numbers and punctuation, so phrases are alphabetic content words.",
                "Nothing is fitted or persisted; the operation records a history entry and a disclosure.",
            ),
            interpretation_rules=(
                "Scores are comparable within one call, never across methods or corpora.",
                "document_frequency shows how many documents contain the phrase; a high corpus score with df=1 is a single-document quirk.",
                "TextRank rewards words that co-occur widely; RAKE rewards long distinctive phrases.",
            ),
            assumptions=(
                "Stopword list matches the corpus language.",
                "Documents are long enough for candidate phrases to exist.",
            ),
            failure_modes=(
                "Boilerplate phrases dominating the corpus ranking.",
                "An empty ranking when stopwords, min_df, and max_phrase_words are jointly too strict.",
                "Single-word rankings on very short documents.",
            ),
            anti_patterns=(
                "Claiming keyphrase precision or recall with no annotated reference.",
                "Using holdout keyphrases to pick features or hyperparameters.",
                "Treating keyphrases and topics as the same product.",
            ),
            worked_example_pattern=(
                "extract_keyphrases(partition='train', method='rake', top_n=20) -> compare with fit_topics term lists.",
            ),
            related_concepts=(
                "nlp-topic-models",
                "nlp-extractive-summarization",
                "nlp-text-normalization",
            ),
        ),
        _note(
            key="nlp-lexicon-sentiment",
            title="Lexicon sentiment: transparent rules with a measurable blind spot",
            summary="A valence lexicon with negation, intensifier, contrast, punctuation and capitalisation rules needs no training data: and reports how often it matched nothing.",
            definition=(
                "Lexicon sentiment sums per-term valence scores, then applies "
                "rules: a negator within a short window flips and damps the "
                "following term, an intensifier scales it with decay by "
                "distance, contrast markers ('but') reweight clauses, "
                "exclamation marks and all-caps add emphasis, and question "
                "marks damp. The sum is squashed into a bounded compound score "
                "and thresholded into positive, negative, or neutral."
            ),
            intuition=(
                "'Not good' is not 'good', and 'VERY BAD!!' is worse than 'bad'. "
                "The rules encode those adjustments explicitly, so you can read "
                "why a document scored the way it did."
            ),
            formal_idea=(
                "compound = sum / sqrt(sum^2 + alpha) with alpha fixed, bounding "
                "the score in (-1, 1). Negation applies a fixed damping factor "
                "with sign flip; intensifiers scale the running term magnitude "
                "with linear decay over token distance."
            ),
            why_it_matters=(
                "It works on day one with no labels, which makes it the right baseline before any supervised sentiment model.",
                "The matched-term rate distinguishes 'genuinely balanced' from 'the lexicon recognised nothing', which a bare neutral share hides.",
                "A supervised classifier fitted on your own labels usually beats it: and the comparison is only meaningful if the baseline is stated.",
            ),
            how_buildml_uses=(
                "analyze_sentiment(backend='lexicon') scores documents with the shipped English lexicon and no install.",
                "backend='supervised' reuses a fitted text classifier's own labels instead of inventing a second scorer.",
                "backend='transformer' delegates to a sentiment checkpoint via buildml[nlp].",
                "compare_to_target reports agreement when the dataset target looks like sentiment labels.",
            ),
            interpretation_rules=(
                "matched_term_rate near zero means the neutral share is ignorance, not balance.",
                "Compound scores are bounded and ordinal, not probabilities.",
                "Agreement against a sentiment-like target is a sanity check, not a validation of the lexicon.",
            ),
            assumptions=(
                "Documents are English; the shipped lexicon covers no other language.",
                "The threshold (default 0.05) suits the register of the corpus.",
            ),
            failure_modes=(
                "Domain jargon carrying sentiment the lexicon has never seen.",
                "Sarcasm and rhetorical questions scored at face value.",
                "Long documents averaging out to neutral because opposing clauses cancel.",
            ),
            anti_patterns=(
                "Reporting lexicon sentiment as ground truth instead of as a baseline.",
                "Applying the English lexicon to a multilingual corpus without running detect_language.",
                "Ignoring matched_term_rate when explaining a large neutral share.",
            ),
            worked_example_pattern=(
                "analyze_sentiment(backend='lexicon') -> fit_text_classifier on labels -> analyze_sentiment(backend='supervised') and compare.",
            ),
            related_concepts=(
                "nlp-language-identification",
                "nlp-document-representation",
                "nlp-rule-vs-statistical-ner",
            ),
        ),
        _note(
            key="nlp-rule-vs-statistical-ner",
            title="Rule entities favour precision; statistical NER favours recall",
            summary="Regex patterns and gazetteers find exactly what you described and nothing else; a trained model generalizes but hallucinates spans.",
            definition=(
                "Entity extraction locates typed spans in text. BuildML's rule "
                "backend applies precision-first regular expressions (dates, "
                "money, percentages, emails, URLs, phone numbers, "
                "identifiers) plus caller-supplied gazetteers matched on whole "
                "words, and resolves overlapping spans by length and pattern "
                "priority. The spaCy backend runs a trained statistical model."
            ),
            intuition=(
                "A rule is a promise: if it matches, the span really is that "
                "type. A model is a guess with better coverage. Which you want "
                "depends on the cost of a false positive."
            ),
            formal_idea=(
                "Rules define a recognizer R with high precision and unknown "
                "recall over the types it encodes, and zero recall over types it "
                "does not. A statistical tagger estimates "
                "p(label | token, context) and has non-zero error on every type."
            ),
            why_it_matters=(
                "Extraction feeding an automated decision usually needs precision, so rules are often the correct engineering answer, not a fallback.",
                "Saying 'the rules cannot find organisations' is more useful than silently returning nothing.",
                "Gazetteers let domain knowledge enter without training data.",
            ),
            how_buildml_uses=(
                "extract_entities(backend='rules') needs no install and reports the pattern source per span.",
                "backend='spacy' requires buildml[nlp-industry] plus a downloaded model, and label names are normalized to BuildML's set.",
                "Overlaps are resolved deterministically so spans never double-count characters.",
            ),
            interpretation_rules=(
                "source names the rule or gazetteer that produced the span, which makes false positives fixable.",
                "label_counts of zero for a type means no pattern covers it, not that the corpus lacks it.",
                "start and end are character offsets into the raw document, so spans can be highlighted exactly.",
            ),
            assumptions=(
                "Documents are plain text with meaningful character offsets.",
                "Gazetteer terms are whole words rather than substrings.",
            ),
            failure_modes=(
                "Gazetteer terms that are common words, producing floods of matches.",
                "Locale-specific date and phone formats outside the shipped patterns.",
                "spaCy backend requested without the model downloaded.",
            ),
            anti_patterns=(
                "Presenting rule output as trained NER performance.",
                "Adding a rule so broad that precision is lost, which defeats the point of the backend.",
                "Assuming spaCy labels mean the same thing as rule labels without checking the mapping.",
            ),
            worked_example_pattern=(
                "extract_entities(backend='rules', gazetteers={'PRODUCT': ['widget-9']}) -> inspect label_counts and source.",
            ),
            related_concepts=(
                "nlp-lexicon-sentiment",
                "nlp-text-normalization",
                "nlp-vs-rag",
            ),
        ),
        _note(
            key="nlp-extractive-summarization",
            title="Extractive summarization selects sentences; it does not write them",
            summary="TextRank and LexRank rank sentences by graph centrality and return the originals in order: and the lead-k baseline is often hard to beat.",
            definition=(
                "Extractive summarization scores the sentences of a document and "
                "returns the highest-scoring ones, unchanged and in their "
                "original order. TextRank builds a sentence graph from token "
                "overlap; LexRank uses TF-IDF cosine similarity with a "
                "similarity floor; lead-k simply takes the first k sentences."
            ),
            intuition=(
                "The most representative sentence is the one most like the rest "
                "of the document. Ranking sentences by that similarity gives a "
                "summary made only of things the author actually wrote."
            ),
            formal_idea=(
                "Build similarity matrix S over sentences, then run PageRank: "
                "s = (1 - d) / n + d * S_norm^T s. Select the top k by score and "
                "emit them in document order. Compression = summary length / "
                "document length."
            ),
            why_it_matters=(
                "Nothing is invented, so an extractive summary cannot state a fact the document does not contain.",
                "Lead-k is a genuinely strong baseline for news-like text; shipping it makes the comparison available instead of assumed.",
                "Without reference summaries no ROUGE score is meaningful, so none is reported.",
            ),
            how_buildml_uses=(
                "summarize_text(method='textrank' | 'lexrank' | 'lead') returns summaries, selected sentence indices, and mean compression.",
                "Sentence splitting is abbreviation-aware so 'Dr. Smith' does not become two sentences.",
                "max_input_sentences bounds the graph size on very long documents.",
            ),
            interpretation_rules=(
                "selected_sentence_indices lets you highlight the summary inside the original document.",
                "mean_compression near 1.0 means the document was already about as short as the requested summary.",
                "Comparing a graph method against 'lead' on your own corpus is the only honest quality claim available here.",
            ),
            assumptions=(
                "Documents contain multiple sentences with detectable boundaries.",
                "Sentence similarity is a reasonable proxy for importance in this genre.",
            ),
            failure_modes=(
                "Single-sentence documents, where every method returns the input.",
                "Lists and tables, where sentence boundaries are meaningless.",
                "Redundant near-identical sentences all scoring highly.",
            ),
            anti_patterns=(
                "Calling extractive output 'generated' or 'abstractive'.",
                "Reporting ROUGE without reference summaries.",
                "Skipping the lead baseline and asserting the graph method is better.",
            ),
            worked_example_pattern=(
                "summarize_text(method='lead', n_sentences=3) then summarize_text(method='textrank', n_sentences=3) and compare on the same documents.",
            ),
            related_concepts=(
                "nlp-keyphrases-vs-topics",
                "nlp-text-normalization",
                "nlp-vs-rag",
            ),
        ),
        _note(
            key="nlp-language-identification",
            title="Language identification and the honest 'und' answer",
            summary="Script ranges plus function-word markers identify language cheaply; short text is reported as undetermined instead of guessed.",
            definition=(
                "Language identification assigns a language code to a document. "
                "The native backend first measures Unicode script shares "
                "(Latin, Cyrillic, Greek, Arabic, Hebrew, Han, Hiragana, "
                "Katakana, Hangul, Devanagari, Thai) and then, within Latin "
                "script, scores discriminative function words. The langdetect "
                "backend delegates to a trained n-gram model."
            ),
            intuition=(
                "Which alphabet is this written in, and which little words does "
                "it use? Those two questions settle most documents without a "
                "model."
            ),
            formal_idea=(
                "score(L) = sum over observed marker words w of weight(L, w), "
                "normalized by token count, with markers down-weighted when they "
                "are shared across languages. Below a minimum character count or "
                "a minimum score, return 'und'."
            ),
            why_it_matters=(
                "Sentiment lexicons, stopword lists, and stemmers are language-specific; applying an English pipeline to a mixed corpus quietly degrades everything.",
                "Refusing to guess on ten characters is more useful than a confident wrong label.",
                "A dominant-language check is one of the cheapest data-quality screens available for text.",
            ),
            how_buildml_uses=(
                "detect_language(backend='native' | 'langdetect') reports per-document codes, counts, dominant language, and the undetermined rate.",
                "profile_text_corpus runs detection as part of the corpus health screen.",
                "The native backend needs no install; langdetect comes with buildml[nlp].",
            ),
            interpretation_rules=(
                "'und' means below the evidence threshold, not 'unknown language'.",
                "A high undetermined rate usually means very short documents rather than exotic languages.",
                "Confidence is a relative marker score, not a calibrated probability.",
            ),
            assumptions=(
                "Documents are predominantly single-language.",
                "The corpus language is among the shipped marker sets for the native backend.",
            ),
            failure_modes=(
                "Code, identifiers, or URLs dominating the token stream.",
                "Closely related languages confused by shared function words.",
                "Code-switched documents receiving one label.",
            ),
            anti_patterns=(
                "Forcing a label on documents below the character threshold.",
                "Running English stopwords and stemming on a corpus that detection flagged as mixed.",
                "Treating the native backend as a general-purpose language classifier for all of Unicode.",
            ),
            worked_example_pattern=(
                "detect_language(partition='all') -> if dominant_language != 'en', reconsider stopword_language and sentiment backend.",
            ),
            related_concepts=(
                "nlp-lexicon-sentiment",
                "nlp-corpus-contamination",
                "nlp-text-normalization",
            ),
        ),
        _note(
            key="nlp-corpus-contamination",
            title="Text contamination: duplicate and near-duplicate documents across a split",
            summary="Text corpora are full of copies, and a holdout document that also appears in train turns evaluation into memorisation: so BuildML measures it and reports it.",
            definition=(
                "Text contamination is the presence of holdout documents that "
                "are identical, or near-identical, to train documents. BuildML "
                "screens for both: exact matches on a normalized fingerprint, "
                "and near-duplicates as pairs whose character n-gram cosine "
                "similarity exceeds a threshold. It also reports the holdout "
                "out-of-vocabulary token rate as the complementary signal."
            ),
            intuition=(
                "If the test set contains the same review twice, once in train, "
                "the model does not need to generalize to score it. Random "
                "splitting cannot detect that, because it only looks at rows."
            ),
            formal_idea=(
                "For train set A and holdout set B, exact overlap is "
                "|{b in B : fingerprint(b) in fingerprint(A)}|. Near-duplicates "
                "are pairs (a, b) with cos(phi(a), phi(b)) >= tau over character "
                "n-gram vectors phi."
            ),
            why_it_matters=(
                "Duplicate documents are the single most common reason a text model's holdout score does not survive deployment.",
                "Reporting contamination instead of silently deduplicating keeps the decision: and its consequences: with the analyst.",
                "Vocabulary drift and contamination are opposite failure signals; seeing both at once tells you which risk you actually have.",
            ),
            how_buildml_uses=(
                "profile_text_corpus reports exact overlap, near-duplicate pairs, the threshold used, duplicate groups, and holdout OOV rate.",
                "Findings are surfaced as warnings on the result and in the Session walkthrough.",
                "Nothing is removed; deduplication stays an explicit choice.",
            ),
            interpretation_rules=(
                "A non-zero exact overlap invalidates the holdout estimate for those rows outright.",
                "Near-duplicate counts depend on the threshold; report the threshold with the number.",
                "A very low holdout OOV rate alongside high near-duplicate counts is the classic contamination signature.",
            ),
            assumptions=(
                "A split exists; without one there is no contamination to measure.",
                "Character n-grams are an adequate similarity proxy for this genre.",
            ),
            failure_modes=(
                "Templated documents flagged as near-duplicates when only the boilerplate matches.",
                "Threshold set so high that paraphrased duplicates slip through.",
                "Very large corpora making exhaustive pair comparison expensive.",
            ),
            anti_patterns=(
                "Deduplicating automatically and reporting a clean score.",
                "Random-splitting a corpus known to contain reposts or syndicated copies.",
                "Quoting a near-duplicate count without the threshold that produced it.",
            ),
            worked_example_pattern=(
                "split(...) -> profile_text_corpus(near_duplicate_threshold=0.9) -> read findings before fit_text_classifier.",
            ),
            related_concepts=(
                "nlp-document-representation",
                "nlp-language-identification",
                "leakage-boundary",
                "rag-eval-contamination",
            ),
        ),
        _note(
            key="nlp-vs-rag",
            title="NLP is not RAG, not text_features, and not Torch fine-tuning",
            summary="Four BuildML surfaces touch text for different reasons; sharing a column does not make them the same product.",
            definition=(
                "buildml.nlp models and analyses a text column on the Session "
                "dataset: classify documents, interpret tokens, fit topics, "
                "extract keyphrases and entities, score sentiment, summarize "
                "extractively, detect language, and profile corpus health. "
                "buildml.rag indexes a document corpus and retrieves chunks for "
                "generation. Session.text_features expands a text column into "
                "tabular columns for a downstream tabular model. The Torch text "
                "path fine-tunes neural networks on text."
            ),
            intuition=(
                "NLP asks 'what does this column say and can I predict from "
                "it?'. RAG asks 'which document answers this question?'. "
                "text_features asks 'how do I get text into my tabular model?'. "
                "Torch asks 'how do I train a network end to end?'."
            ),
            formal_idea=(
                "Distinct state, distinct artifacts: NlpTextPlan and "
                "buildml.nlp_bundle.v1 versus RAG indexes and "
                "buildml.rag_bundle, versus TextFeaturePlan inside the "
                "preprocessing pipeline, versus Torch trainer bundles."
            ),
            why_it_matters=(
                "Choosing the wrong surface costs correctness, not just convenience: RAG has no holdout metric for classification, and text_features has no token attribution.",
                "Separate bundles prevent one artifact from silently standing in for another.",
                "Naming the boundary keeps optional dependencies honest: the core NLP path needs no transformer stack.",
            ),
            how_buildml_uses=(
                "Separate packages: buildml.nlp, buildml.rag, buildml.preprocess.text, buildml.dl.",
                "Separate Session state: _nlp_text_plan versus RAG state versus the preprocessing pipeline.",
                "Separate bundle formats, each validated on load.",
            ),
            interpretation_rules=(
                "Use fit_text_classifier when the label is on the row and the text is the evidence.",
                "Use rag_ingest_corpus when the answer lives in a document you must find and cite.",
                "Use text_features when text is one signal among many tabular columns.",
                "Use the Torch text path when you need to update model weights on your own data.",
            ),
            assumptions=("Product surfaces stay distinct and independently versioned.",),
            failure_modes=(
                "Expecting checkpoint_load to restore an NLP plan.",
                "Expecting a RAG bundle to load as an NLP bundle.",
                "Assuming buildml[nlp] extras are required for the core text classifier.",
            ),
            anti_patterns=(
                "Routing document classification through RAG retrieval.",
                "Calling NLP 'RAG for columns'.",
                "Using text_features and then claiming token-level explanations of the tabular model.",
            ),
            worked_example_pattern=(
                "fit_text_classifier(...) for document classification; rag_ingest_corpus(...) for grounded answers; text_features(...) for tabular expansion.",
            ),
            related_concepts=(
                "nlp-document-representation",
                "nlp-bundle-boundary",
                "rag-chunk-index-boundary",
                "text-features",
            ),
        ),
        _note(
            key="nlp-bundle-boundary",
            title="NLP bundle vs Session checkpoint",
            summary="buildml.nlp_bundle.v1 stores the fitted vectorizer, head, and topic model; Session checkpoints store workflow state and do not embed them.",
            definition=(
                "save_nlp_bundle writes meta.json plus joblib payloads for the "
                "text plan and the topic plan. checkpoint_save records the "
                "Session workflow without the NLP vectorizer, head, or "
                "decomposition. The two artifacts are complementary."
            ),
            intuition=(
                "Reload the model with load_nlp_bundle; reload the workflow with "
                "checkpoint_load. Expecting one to do the other loses work."
            ),
            formal_idea=(
                "Artifact separation: the learner bundle is orthogonal to the "
                "Session checkpoint, and each is validated by its own format tag."
            ),
            why_it_matters=(
                "The normalization plan travels inside the bundle, so a reloaded model preprocesses text exactly as it was fitted.",
                "Prevents silent loss of a fitted vocabulary across restarts.",
                "Keeps NLP bundles distinguishable from RAG, CBR, classical, and Torch bundles on load.",
            ),
            how_buildml_uses=(
                "Session.save_nlp_bundle / load_nlp_bundle; format is validated as buildml.nlp_bundle.v1.",
                "A bundle may carry a text plan, a topic plan, or both.",
                "Loading clears stale fit, eval, predict, and interpret results so nothing misattributes to the new plan.",
            ),
            interpretation_rules=(
                "Confirm meta.json format == buildml.nlp_bundle.v1 before trusting a directory.",
                "After load, evaluate on holdout rather than assuming the recorded metrics still apply to this dataset.",
            ),
            assumptions=(
                "Writable path; the reloading Session exposes the same text column name.",
            ),
            failure_modes=(
                "Expecting checkpoint_load to restore NlpTextPlan.",
                "Loading a bundle into a Session whose text column has a different name.",
                "A partially written bundle directory.",
            ),
            anti_patterns=(
                "Treating RAG bundles and NLP bundles as interchangeable.",
                "Shipping a fitted vectorizer without its normalization plan.",
            ),
            worked_example_pattern=(
                "fit_text_classifier(...) -> save_nlp_bundle(path) -> new Session -> load_nlp_bundle(path) -> evaluate_text_classifier(partition='test').",
            ),
            related_concepts=(
                "nlp-vs-rag",
                "nlp-text-normalization",
                "checkpoint-integrity",
                "leakage-boundary",
            ),
        ),
    )
}
