# ruff: noqa: E501
"""Beginner layers for self-supervised representation learning."""

from __future__ import annotations

from buildml.explain.beginner._builder import CORE, BeginnerLayer, _index, _layer

SELFSUPERVISED_BEGINNER: dict[str, BeginnerLayer] = _index(
    _layer(
        "ssl-pretext-then-head",
        plain=(
            "Self-supervised learning invents a training task out of the data itself so no human labels are "
            "needed. You hide part of each row and train a model to reconstruct it. That model learns a "
            "useful compressed representation, which you then freeze and put a small labelled model on top of."
        ),
        analogy=(
            "Learning a language by filling in blanked-out words in books. Nobody grades you, but you end up "
            "understanding the language: and then a short course teaches you the specific job you need it for."
        ),
        steps=(
            "Pretrain: hide random parts of the training features and train an encoder to reconstruct them. Labels are ignored completely.",
            "Freeze the encoder so its weights stop changing.",
            "Transform your rows through the encoder to get embeddings.",
            "Attach a small supervised head and train it on your labelled training rows only.",
            "Evaluate the whole frozen stack on labelled holdout rows.",
        ),
        use=(
            "When unlabelled rows massively outnumber labelled ones and the features have exploitable structure.",
            "When you want one reusable representation feeding several downstream tasks.",
        ),
        avoid=(
            "Do not bother on small, tidy tabular data with plenty of labels: a gradient-boosting model will beat this and take a fraction of the effort.",
            "Do not pretrain on validation or test rows; the encoder is learned information and leaks like any other fitted object.",
        ),
        myths=(
            (
                "Self-supervised means no supervision anywhere.",
                "The pretext task needs no human labels. The downstream head still needs real labels: you just need far fewer of them.",
            ),
            (
                "A good reconstruction loss means a good representation.",
                "Reconstruction can be dominated by easy, high-variance columns. Only downstream performance tells you whether the embedding is useful.",
            ),
        ),
        example=(
            "session.fit_ssl_pretext(method='masked_tabular', mask_rate=0.3, epochs=50)",
            "session.finetune_ssl_head(estimator=LogisticRegression(max_iter=1000))",
            "session.evaluate_ssl(partition='validation')",
        ),
        check=(
            "Does the frozen-encoder-plus-head beat a plain model on your raw features?",
            "How many labelled rows do you actually have?",
        ),
        tools=("fit_ssl_pretext", "transform_ssl", "finetune_ssl_head", "evaluate_ssl"),
        terms=("self-supervised", "embedding", "fine-tuning", "supervised"),
        difficulty=CORE,
    ),
    _layer(
        "ssl-masked-tabular",
        plain=(
            "The tabular version of the hide-and-reconstruct trick. BuildML randomly blanks out some feature "
            "values in each training row and trains a small neural network to guess them back. The narrow "
            "middle layer of that network becomes your embedding."
        ),
        analogy=(
            "Practising a crossword where random letters are removed. To fill them in you have to learn how "
            "the words relate: and that understanding is the thing you actually keep."
        ),
        steps=(
            "Choose a mask rate: what fraction of values to hide per row. Around 0.15 to 0.3 is typical.",
            "The network sees the masked row and tries to reproduce the original.",
            "Its bottleneck layer is forced to hold a compact summary of the row.",
            "After training, discard the reconstruction output and keep the bottleneck as your embedding.",
            "Export those embeddings with `transform_ssl` for any downstream model.",
        ),
        use=(
            "On wide tabular data where columns are genuinely related, so reconstruction requires learning structure.",
            "When you want a fixed-size numeric representation of a row for clustering, similarity, or a downstream head.",
        ),
        avoid=(
            "Do not use it on data whose columns are mutually independent; there is nothing to learn from reconstruction.",
            "Do not set the mask rate so high that the network has nothing to work from, or so low that copying the input suffices.",
        ),
        myths=(
            (
                "A bigger bottleneck gives a better embedding.",
                "A bottleneck as wide as the input can just copy the row and learn nothing. The constraint is what creates the representation.",
            ),
            (
                "Masked pretraining works the same on tabular data as on text.",
                "Text has enormous sequential redundancy that makes the task rich. Tabular columns are often far less predictable from each other, so gains are smaller and less certain.",
            ),
        ),
        example=(
            "session.fit_ssl_pretext(",
            "    method='masked_tabular', mask_rate=0.25,",
            "    embedding_dim=32, epochs=50, random_state=0,",
            ")",
            "session.transform_ssl()   # embedding columns join the frame",
        ),
        check=(
            "Is your embedding dimension meaningfully smaller than your feature count?",
            "Can any of your columns be predicted from the others at all?",
        ),
        tools=("fit_ssl_pretext", "transform_ssl", "evaluate_ssl", "fit_clusters"),
        terms=("self-supervised", "embedding", "neural network", "epoch"),
        difficulty=CORE,
    ),
    _layer(
        "ssl-vs-backbone-transfer",
        plain=(
            "Two different ways to reuse learned representations, and BuildML keeps them on separate "
            "surfaces. Tabular self-supervision trains an encoder on *your* rows from scratch. Backbone "
            "transfer downloads a model someone else pretrained on images, audio, or speech and adapts it."
        ),
        analogy=(
            "Teaching yourself a subject from the material in front of you, versus hiring someone who "
            "already spent years studying it and briefing them on your specifics."
        ),
        steps=(
            "Identify your data type. Tabular rows point to the SSL path; images, audio, or speech point to backbones.",
            "For tabular, use `fit_ssl_pretext`: the encoder is trained on your data alone.",
            "For images or audio, use `load_pretrained_backbone` and then `attach_backbone_head`.",
            "Decide whether to freeze the backbone or fine-tune it; freezing needs far less data.",
            "Evaluate either path on labelled holdout rows the same way.",
        ),
        use=(
            "SSL when your data is tabular and no relevant pretrained model exists.",
            "Backbone transfer when your data is a common modality where enormous pretrained models are available.",
        ),
        avoid=(
            "Do not expect an image backbone to help with a tabular table; the pretraining domain has to be related.",
            "Do not fine-tune a large backbone on a few hundred labelled rows without freezing most of it: you will overfit spectacularly.",
        ),
        myths=(
            (
                "Pretrained models exist for everything, so training an encoder is obsolete.",
                "For arbitrary business tables there is no useful pretrained model. Your schema is unique to you.",
            ),
            (
                "Fine-tuning is always better than freezing.",
                "Fine-tuning needs enough labelled data to justify moving millions of parameters. Below that, a frozen backbone plus a small head wins.",
            ),
        ),
        example=(
            "# tabular:",
            "session.fit_ssl_pretext(method='masked_tabular', epochs=50)",
            "# images / audio / speech:",
            "session.load_pretrained_backbone(name='resnet18', freeze=True)",
            "session.attach_backbone_head(num_classes=5)",
        ),
        check=(
            "Is there a pretrained model whose training domain resembles your data?",
            "How many labelled rows do you have per class?",
        ),
        tools=("fit_ssl_pretext", "load_pretrained_backbone", "attach_backbone_head", "finetune_ssl_head"),
        terms=("self-supervised", "backbone", "transfer learning", "fine-tuning"),
        difficulty=CORE,
    ),
    _layer(
        "ssl-bundle-boundary",
        plain=(
            "The pretrained encoder: and optionally the head you attached: saves as a self-supervised "
            "bundle. It is not a Session checkpoint and not a Torch trainer bundle; those store different "
            "things with different contracts."
        ),
        analogy=(
            "The trained interpreter, the transcript of the meeting, and the meeting-room booking are three "
            "separate records. Only one of them can translate for you tomorrow."
        ),
        steps=(
            "Pretrain an encoder so an SSL plan exists.",
            "Optionally attach and fit a head.",
            "Call `save_ssl_bundle(path)` to store the encoder and, if present, the head.",
            "Reload with `load_ssl_bundle(path)` and call `transform_ssl` on new rows.",
            "Keep checkpoints and Torch trainer bundles separately; each answers a different question.",
        ),
        use=(
            "When the representation is reusable across several downstream projects.",
            "When a scheduled job needs to embed new rows without redoing pretraining.",
        ),
        avoid=(
            "Do not load a Torch trainer bundle expecting the SSL encoder; they are distinct artifact types.",
            "Do not assume the bundle carries the downstream head unless you fitted and saved one.",
        ),
        myths=(
            (
                "The Torch bundle and the SSL bundle overlap enough to be interchangeable.",
                "The trainer bundle stores a supervised training run. The SSL bundle stores a pretext encoder. Loading the wrong one fails at load time by design.",
            ),
            (
                "Embeddings can be saved instead of the encoder.",
                "Saved embeddings only cover the rows you already had. The encoder is what lets you embed a row you have never seen.",
            ),
        ),
        example=(
            "session.save_ssl_bundle('artifacts/tab-encoder')",
            "job = Session.ingest(new_frame).load_ssl_bundle('artifacts/tab-encoder')",
            "job.transform_ssl()   # embeddings for previously unseen rows",
        ),
        check=(
            "Does your bundle include the head, or only the encoder?",
            "Which artifact would you reload to embed a brand-new row?",
        ),
        tools=("save_ssl_bundle", "load_ssl_bundle", "transform_ssl", "save_torch_bundle"),
        terms=("bundle", "checkpoint", "embedding", "self-supervised"),
        difficulty=CORE,
    ),
)

__all__ = ["SELFSUPERVISED_BEGINNER"]
