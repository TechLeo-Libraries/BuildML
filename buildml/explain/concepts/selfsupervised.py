# ruff: noqa: E501
"""Self-supervised learning concept notes."""

from __future__ import annotations

from buildml.explain.concepts._builder import _note
from buildml.explain.schemas import ConceptNote

SELFSUPERVISED_NOTES: dict[str, ConceptNote] = {
    note.key: note
    for note in (
        _note(
            key="ssl-pretext-then-head",
            title="SSL pretext then supervised head",
            summary="Pretrain a representation on train features (labels ignored), freeze it, then attach a head on labeled train rows only.",
            definition=(
                "Self-supervised hooks learn f_pretext on X_train, export "
                "representations z = enc(X), then fit a supervised head g on labeled "
                "(z_train, y_train) without updating the encoder during head fit."
            ),
            intuition=(
                "First practice describing the worksheets without looking at answers; "
                "then learn a small decoder from the scarce graded pages."
            ),
            formal_idea=(
                "min_θ L_pretext(X_train; θ); freeze enc_θ; "
                "min_φ L_sup(g_φ(enc_θ(X_labeled)), y_labeled)."
            ),
            why_it_matters=(
                "Separating pretext from head keeps unlabeled abundance useful without "
                "leaking holdout labels into representation learning.",
            ),
            how_buildml_uses=(
                "Session.fit_ssl_pretext → finetune_ssl_head → evaluate_ssl.",
                "transform_ssl can attach embedding columns for classical Session.fit.",
            ),
            interpretation_rules=(
                "Reconstruction MAE is a pretext diagnostic, not predictive utility.",
                "Read n_labeled_train / n_unlabeled_skipped on the head fit.",
            ),
            assumptions=("Train features are numeric and imputed/scaled as needed.",),
            failure_modes=(
                "Updating the encoder on test during 'fine-tune'.",
                "Treating train reconstruction error as accuracy.",
            ),
            anti_patterns=(
                "Calling reconstruction MAE a classification score.",
            ),
            worked_example_pattern=(
                "fit_ssl_pretext(latent_dim=16) → finetune_ssl_head() → evaluate_ssl().",
            ),
            related_concepts=("ssl-masked-tabular", "semisupervised-label-missingness"),
        ),
        _note(
            key="ssl-masked-tabular",
            title="Masked tabular pretext",
            summary="A compact complete surface: mask features, reconstruct with an MLP, export the bottleneck as embeddings.",
            definition=(
                "Masked tabular SSL randomly masks a fraction of train features, "
                "trains a multi-output MLP to reconstruct the original vector, and "
                "exports latent-layer activations as the representation."
            ),
            intuition=(
                "Cover some cells, learn to fill them from the rest, then keep the "
                "internal summary vector as features for a small classifier."
            ),
            formal_idea=(
                "Sample mask M; minimize ||g_ψ(h_θ(x ⊙ (1-M) ⊕ fill)) - x||; "
                "representation z = h_θ(x)."
            ),
            why_it_matters=(
                "Ships a real, tested pretext without pretending to be a contrastive FM zoo.",
            ),
            how_buildml_uses=(
                "method='masked_tabular' on Session.fit_ssl_pretext (core sklearn path).",
            ),
            interpretation_rules=(
                "Contrastive SimCLR/MoCo and BERT-from-scratch are out of scope here.",
            ),
            assumptions=("Enough train rows for a small MLP; mask_ratio ∈ (0,1).",),
            failure_modes=("Mask ratio 0 or 1; colliding representation column names.",),
            anti_patterns=(
                "Shipping stub catalogs of many pretext tasks without a working path.",
            ),
            worked_example_pattern=("fit_ssl_pretext(mask_ratio=0.2, latent_dim=8).",),
            related_concepts=("ssl-pretext-then-head", "ssl-vs-backbone-transfer"),
        ),
        _note(
            key="ssl-vs-backbone-transfer",
            title="Tabular SSL vs Torch backbone transfer",
            summary="Tabular masked pretext is core Session SSL; vision/audio/speech freeze/finetune stays on load_pretrained_backbone / attach_backbone_head.",
            definition=(
                "BuildML separates tabular self-supervised hooks from optional Torch "
                "pretrained backbone transfer. The zoo path loads published weights "
                "(or mock/none) and attaches a head; it is not tabular masked AE training."
            ),
            intuition=(
                "One toolkit learns a small table encoder from your rows; another "
                "reuses a camera/microphone teacher and only trains the last stickers."
            ),
            formal_idea=(
                "Tabular: fit enc on X_train. Backbone: load f_pretrained; train head "
                "on frozen or lightly finetuned features."
            ),
            why_it_matters=(
                "Keeps core light while still offering honest Torch transfer under extras.",
            ),
            how_buildml_uses=(
                "Session.fit_ssl_pretext vs Session.load_pretrained_backbone + attach_backbone_head.",
            ),
            interpretation_rules=(
                "Do not claim the tabular path trains Whisper/BERT foundations.",
            ),
            assumptions=("Torch/speech extras installed for backbone paths.",),
            failure_modes=("Calling backbone transfer 'tabular SSL' without disclosure.",),
            anti_patterns=(
                "Reimplementing the zoo inside buildml.selfsupervised as duplicate stubs.",
            ),
            worked_example_pattern=(
                "Tables → fit_ssl_pretext; images → load_pretrained_backbone('resnet18').",
            ),
            related_concepts=("ssl-pretext-then-head", "ssl-masked-tabular"),
        ),
        _note(
            key="ssl-bundle-boundary",
            title="Self-supervised bundle boundary",
            summary="buildml.selfsupervised_bundle.v1 stores the encoder (+ optional head); checkpoints and Torch trainer bundles are different artifacts.",
            definition=(
                "An SSL bundle persists the train-fitted SelfSupervisedPlan and optional "
                "SSLHeadPlan. Session checkpoints and Torch trainer bundles do not "
                "substitute for it."
            ),
            intuition=(
                "Saving your homework resume is not the same as shipping the learned encoder."
            ),
            formal_idea=(
                "checkpoint_load ↛ SSL encoder; load_ssl_bundle ↛ dataset rows; "
                "torch_bundle ↛ tabular MaskedTabularEncoder."
            ),
            why_it_matters=("Artifact confusion drops weights or pretends loaders restored.",),
            how_buildml_uses=("save_ssl_bundle / load_ssl_bundle.",),
            interpretation_rules=("Read meta.json format buildml.selfsupervised_bundle.v1.",),
            assumptions=("Feature contract still matches at load time.",),
            failure_modes=("Expecting checkpoint_load to restore SelfSupervisedPlan.",),
            anti_patterns=("Assuming torch_bundle load restores masked tabular encoders.",),
            worked_example_pattern=(
                "session.save_ssl_bundle(path); other.load_ssl_bundle(path).",
            ),
            related_concepts=("ssl-pretext-then-head",),
        ),
    )
}
