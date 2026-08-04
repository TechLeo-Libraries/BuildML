"""Declarative Session namespaced-facade registry.

Each domain exposes ``session.<attr>.*`` bindings that delegate to
existing flat Session methods. Regenerate with
``python scripts/generate_facade_registry.py``.
"""

from __future__ import annotations

from typing import Any, Final, TypedDict


class DomainFacadeSpec(TypedDict):
    """Typed registry row for one namespaced Session facade."""

    mixin_key: str
    tier: str  # core | domain | experimental
    warn_flat: bool
    bindings: dict[str, str]  # facade_name -> flat Session method


DOMAIN_FACADES: Final[dict[str, DomainFacadeSpec]] = {
    'active_learning': {
        "mixin_key": 'activelearning',
        "tier": 'domain',
        "warn_flat": True,
        "bindings": {
            'capability_matrix': 'activelearning_capability_matrix',
            'eval_result': 'activelearning_eval_result',
            'evaluate': 'evaluate_active_learning',
            'fit': 'fit_active_learner',
            'fit_result': 'activelearning_fit_result',
            'label_result': 'activelearning_label_result',
            'label_rows': 'label_rows',
            'load_bundle': 'load_active_learning_bundle',
            'plan': 'activelearning_plan',
            'query_result': 'activelearning_query_result',
            'save_bundle': 'save_active_learning_bundle',
            'suggest_query': 'suggest_query',
        },
    },
    'ai': {
        "mixin_key": 'ai',
        "tier": 'experimental',
        "warn_flat": True,
        "bindings": {
            'advisor': 'ai_advisor',
            'configure': 'ai_configure',
            'dry_run': 'ai_dry_run',
            'egress_preview': 'ai_egress_preview',
            'execute': 'ai_execute',
            'load_transcript': 'load_ai_transcript',
            'plan': 'ai_plan',
            'result': 'ai_result',
            'run_autonomous': 'ai_run_autonomous',
            'run_plan': 'ai_run_plan',
            'save_transcript': 'save_ai_transcript',
            'status': 'ai_status',
            'transcript': 'ai_transcript',
        },
    },
    'anomaly': {
        "mixin_key": 'anomaly',
        "tier": 'domain',
        "warn_flat": True,
        "bindings": {
            'capability_matrix': 'anomaly_capability_matrix',
            'eval_result': 'anomaly_eval_result',
            'evaluate': 'evaluate_anomaly',
            'fit': 'fit_anomaly',
            'fit_result': 'anomaly_fit_result',
            'load_bundle': 'load_anomaly_bundle',
            'plan': 'anomaly_plan',
            'save_bundle': 'save_anomaly_bundle',
            'score': 'score_anomalies',
            'score_result': 'anomaly_score_result',
            'tune_threshold': 'tune_anomaly_threshold',
        },
    },
    'audit': {
        "mixin_key": 'workflow',
        "tier": 'core',
        "warn_flat": False,
        "bindings": {
            'describe_method': 'describe_method',
            'dry_run': 'dry_run',
            'explain': 'explain',
            'history': 'history',
            'last_dry_run': 'last_dry_run',
            'last_summary': 'last_history_summary',
            'last_walkthrough': 'last_walkthrough',
            'learn': 'learn',
            'list_active_domains': 'list_active_domains',
            'list_capabilities': 'list_capabilities',
            'summarize': 'summarize_history',
            'walkthrough': 'walkthrough',
            'workflow': 'workflow',
        },
    },
    'automl': {
        "mixin_key": 'automl',
        "tier": 'domain',
        "warn_flat": True,
        "bindings": {
            'capability_matrix': 'automl_capability_matrix',
            'evaluate': 'evaluate_automl',
            'load_bundle': 'load_automl_bundle',
            'plan': 'automl_plan',
            'result': 'automl_result',
            'run': 'run_automl',
            'save_bundle': 'save_automl_bundle',
        },
    },
    'causal': {
        "mixin_key": 'causal',
        "tier": 'domain',
        "warn_flat": True,
        "bindings": {
            'assumptions': 'causal_assumptions',
            'capability_matrix': 'causal_capability_matrix',
            'declare_assumptions': 'declare_causal_assumptions',
            'estimate': 'estimate_causal',
            'estimate_result': 'causal_estimate_result',
            'eval_result': 'causal_eval_result',
            'evaluate': 'evaluate_causal',
            'fit': 'fit_causal',
            'fit_result': 'causal_fit_result',
            'load_bundle': 'load_causal_bundle',
            'plan': 'causal_plan',
            'refute': 'refute_causal',
            'refute_result': 'causal_refute_result',
            'save_bundle': 'save_causal_bundle',
        },
    },
    'cbr': {
        "mixin_key": 'cbr',
        "tier": 'domain',
        "warn_flat": True,
        "bindings": {
            'capability_matrix': 'cbr_capability_matrix',
            'eval_result': 'cbr_eval_result',
            'evaluate': 'evaluate_cbr',
            'fit': 'fit_cbr',
            'fit_result': 'cbr_fit_result',
            'load_bundle': 'load_cbr_bundle',
            'plan': 'cbr_plan',
            'predict': 'predict_cbr',
            'predict_result': 'cbr_predict_result',
            'retain': 'retain_cbr',
            'retain_result': 'cbr_retain_result',
            'retrieve': 'retrieve_cases',
            'retrieve_result': 'cbr_retrieve_result',
            'save_bundle': 'save_cbr_bundle',
        },
    },
    'classical': {
        "mixin_key": 'classical',
        "tier": 'core',
        "warn_flat": False,
        "bindings": {
            'calibration': 'calibration',
            'compare_models': 'compare_models',
            'cv_score': 'cv_score',
            'error_slices': 'error_slices',
            'eval_plots': 'eval_plots',
            'evaluate': 'evaluate',
            'evolutionary_search': 'evolutionary_search',
            'explain_shap': 'explain_shap',
            'feature_importance': 'feature_importance',
            'fit': 'fit',
            'fit_result': 'fit_result',
            'grid_search': 'grid_search',
            'last_cv': 'last_cv',
            'last_nested_cv': 'last_nested_cv',
            'last_plot_board': 'last_plot_board',
            'last_search': 'last_search',
            'learning_curve': 'learning_curve',
            'load_model': 'load_model',
            'load_pipeline': 'load_pipeline',
            'model_card': 'model_card',
            'nested_cv_score': 'nested_cv_score',
            'optuna_search': 'optuna_search',
            'predict': 'predict',
            'predict_from_pipeline': 'predict_from_pipeline',
            'prepare_design_matrix': 'prepare_design_matrix',
            'randomized_search': 'randomized_search',
            'save_model': 'save_model',
            'save_pipeline': 'save_pipeline',
            'tune_threshold': 'tune_threshold',
        },
    },
    'data': {
        "mixin_key": 'data',
        "tier": 'core',
        "warn_flat": False,
        "bindings": {
            'assert_can_fit': 'assert_can_fit',
            'checkpoint_load': 'checkpoint_load',
            'checkpoint_save': 'checkpoint_save',
            'close_native': 'close_native',
            'group_split': 'group_split',
            'head': 'head',
            'ingest': 'ingest',
            'ingest_report': 'ingest_report',
            'inject_split': 'inject_split',
            'metadata': 'metadata',
            'partition': 'partition',
            'reattach': 'reattach',
            'reattach_result': 'reattach_result',
            'set_roles': 'set_roles',
            'split': 'split',
            'split_plan': 'split_plan',
            'sync_native': 'sync_native',
            'time_split': 'time_split',
            'to_engine': 'to_engine',
            'to_pandas': 'to_pandas',
            'to_parquet': 'to_parquet',
            'with_engine': 'with_engine',
            'with_mode': 'with_mode',
        },
    },
    'decision': {
        "mixin_key": 'decision',
        "tier": 'domain',
        "warn_flat": True,
        "bindings": {
            'apply': 'apply_decisions',
            'apply_result': 'decision_apply_result',
            'capability_matrix': 'decision_capability_matrix',
            'eval_result': 'decision_eval_result',
            'evaluate': 'evaluate_decisions',
            'fit': 'fit_decision_policy',
            'fit_result': 'decision_fit_result',
            'load_bundle': 'load_decision_bundle',
            'optimize_capability_matrix': 'optimize_capability_matrix',
            'plan': 'decision_plan',
            'save_bundle': 'save_decision_bundle',
        },
    },
    'dl': {
        "mixin_key": 'dl',
        "tier": 'domain',
        "warn_flat": True,
        "bindings": {
            'asr_eval': 'dl_asr_eval',
            'attach_head': 'attach_backbone_head',
            'backbone': 'dl_backbone',
            'backbone_head': 'dl_backbone_head',
            'capability_matrix': 'dl_capability_matrix',
            'cross_validate': 'cross_validate_torch',
            'cv_result': 'dl_cv_result',
            'ddp_result': 'dl_ddp_result',
            'domain_adapt_speech': 'domain_adapt_speech_torch',
            'emit_k8s_ddp': 'emit_k8s_ddp_job',
            'emit_k8s_serve': 'emit_k8s_serve_deployment',
            'evaluate': 'evaluate_torch',
            'evaluate_asr': 'evaluate_asr',
            'export': 'export_torch',
            'export_result': 'dl_export_result',
            'fit': 'fit_torch',
            'fit_ddp': 'fit_torch_ddp',
            'fit_speech': 'fit_speech_torch',
            'load_backbone': 'load_pretrained_backbone',
            'load_bundle': 'load_torch_bundle',
            'make_audio_loaders': 'make_audio_multimodal_torch_loaders',
            'make_image_loaders': 'make_image_multimodal_torch_loaders',
            'make_loaders': 'make_torch_loaders',
            'make_multimodal_loaders': 'make_multimodal_torch_loaders',
            'make_speech_loaders': 'make_speech_torch_loaders',
            'make_text_loaders': 'make_text_torch_loaders',
            'nested_cv': 'nested_cv_torch',
            'nested_cv_result': 'dl_nested_cv_result',
            'pack_torchserve': 'pack_torchserve',
            'prepare_tensorrt': 'prepare_tensorrt_export',
            'refuse_speech_pretrain': 'refuse_speech_foundation_pretrain',
            'save_bundle': 'save_torch_bundle',
            'search': 'search_torch',
            'search_result': 'dl_search_result',
            'serve': 'serve_bundle',
            'speech_result': 'dl_speech_result',
            'train_result': 'dl_train_result',
            'training_curve': 'torch_training_curve',
            'transcribe': 'transcribe_speech',
        },
    },
    'ensemble': {
        "mixin_key": 'ensemble',
        "tier": 'domain',
        "warn_flat": True,
        "bindings": {
            'capability_matrix': 'ensemble_capability_matrix',
            'evaluate': 'evaluate_ensemble',
            'fit_blending': 'fit_blending',
            'fit_result': 'ensemble_fit_result',
            'fit_stacking': 'fit_stacking',
            'fit_voting': 'fit_voting',
            'load_bundle': 'load_ensemble_bundle',
            'plan': 'ensemble_plan',
            'save_bundle': 'save_ensemble_bundle',
        },
    },
    'explore': {
        "mixin_key": 'eda',
        "tier": 'core',
        "warn_flat": False,
        "bindings": {
            'app': 'eda_app',
            'last_report': 'last_eda',
            'open_dashboard': 'open_eda_dashboard',
            'run': 'eda',
        },
    },
    'fairness': {
        "mixin_key": 'fairness',
        "tier": 'domain',
        "warn_flat": True,
        "bindings": {
            'capability_matrix': 'fairness_capability_matrix',
            'evaluate': 'evaluate_fairness',
            'attach_to_last_eval': 'attach_fairness_to_last_eval',
            'suggest_thresholds': 'suggest_fairness_thresholds',
            'suggest_reweighing': 'suggest_fairness_reweighing',
            'last_report': 'last_fairness',
        },
    },
    'federated': {
        "mixin_key": 'federated',
        "tier": 'domain',
        "warn_flat": True,
        "bindings": {
            'capability_matrix': 'federated_capability_matrix',
            'eval_result': 'federated_eval_result',
            'evaluate': 'evaluate_federated',
            'export_round_history': 'export_round_history',
            'fit': 'fit_federated',
            'fit_result': 'federated_fit_result',
            'load_bundle': 'load_federated_bundle',
            'plan': 'federated_plan',
            'predict': 'predict_federated',
            'predict_result': 'federated_predict_result',
            'save_bundle': 'save_federated_bundle',
        },
    },
    'forecast': {
        "mixin_key": 'forecast',
        "tier": 'domain',
        "warn_flat": True,
        "bindings": {
            'capability_matrix': 'forecast_capability_matrix',
            'eval_result': 'forecast_eval_result',
            'evaluate': 'evaluate_forecast',
            'fit': 'fit_forecast',
            'fit_result': 'forecast_fit_result',
            'generate': 'generate_forecast',
            'generate_result': 'forecast_generate_result',
            'load_bundle': 'load_forecast_bundle',
            'plan': 'forecast_plan',
            'save_bundle': 'save_forecast_bundle',
        },
    },
    'graph': {
        "mixin_key": 'graph',
        "tier": 'domain',
        "warn_flat": True,
        "bindings": {
            'capability_matrix': 'graph_capability_matrix',
            'eval_result': 'graph_eval_result',
            'evaluate': 'evaluate_graph',
            'fit': 'fit_graph',
            'fit_result': 'graph_fit_result',
            'load_bundle': 'load_graph_bundle',
            'plan': 'graph_plan',
            'predict': 'predict_graph',
            'predict_result': 'graph_predict_result',
            'save_bundle': 'save_graph_bundle',
            'set_spec': 'set_graph',
            'spec': 'graph_spec',
        },
    },
    'kg': {
        "mixin_key": 'kg',
        "tier": 'domain',
        "warn_flat": True,
        "bindings": {
            'capability_matrix': 'kg_capability_matrix',
            'eval_result': 'kg_eval_result',
            'evaluate': 'evaluate_kg',
            'fit': 'fit_kg',
            'fit_result': 'kg_fit_result',
            'load_bundle': 'load_kg_bundle',
            'plan': 'kg_plan',
            'predict_links': 'predict_links',
            'predict_result': 'kg_predict_result',
            'query': 'query_kg',
            'query_result': 'kg_query_result',
            'save_bundle': 'save_kg_bundle',
            'score_result': 'kg_score_result',
            'score_triples': 'score_triples',
        },
    },
    'metalearning': {
        "mixin_key": 'metalearning',
        "tier": 'experimental',
        "warn_flat": True,
        "bindings": {
            'adapt': 'adapt_to_task',
            'adapt_result': 'metalearning_adapt_result',
            'capability_matrix': 'metalearning_capability_matrix',
            'eval_result': 'metalearning_eval_result',
            'evaluate': 'evaluate_metalearning',
            'fit': 'fit_metalearning',
            'fit_result': 'metalearning_fit_result',
            'load_bundle': 'load_metalearning_bundle',
            'plan': 'metalearning_plan',
            'save_bundle': 'save_metalearning_bundle',
        },
    },
    'multitask': {
        "mixin_key": 'multitask',
        "tier": 'domain',
        "warn_flat": True,
        "bindings": {
            'capability_matrix': 'multitask_capability_matrix',
            'eval_result': 'multitask_eval_result',
            'evaluate': 'evaluate_multitask',
            'fit': 'fit_multitask',
            'fit_result': 'multitask_fit_result',
            'load_bundle': 'load_multitask_bundle',
            'plan': 'multitask_plan',
            'predict': 'predict_multitask',
            'predict_result': 'multitask_predict_result',
            'save_bundle': 'save_multitask_bundle',
        },
    },
    'nlp': {
        "mixin_key": 'nlp',
        "tier": 'domain',
        "warn_flat": True,
        "bindings": {
            'analyze_sentiment': 'analyze_sentiment',
            'assign_topics': 'assign_topics',
            'capability_matrix': 'nlp_capability_matrix',
            'detect_language': 'detect_language',
            'entity_result': 'nlp_entity_result',
            'eval_result': 'nlp_eval_result',
            'evaluate': 'evaluate_text_classifier',
            'extract_entities': 'extract_entities',
            'extract_keyphrases': 'extract_keyphrases',
            'fit_classifier': 'fit_text_classifier',
            'fit_result': 'nlp_fit_result',
            'fit_topics': 'fit_topics',
            'interpret': 'interpret_text_prediction',
            'interpret_result': 'nlp_interpret_result',
            'keyphrase_result': 'nlp_keyphrase_result',
            'language_result': 'nlp_language_result',
            'load_bundle': 'load_nlp_bundle',
            'predict': 'predict_text',
            'predict_result': 'nlp_predict_result',
            'profile_corpus': 'profile_text_corpus',
            'profile_result': 'nlp_profile_result',
            'save_bundle': 'save_nlp_bundle',
            'sentiment_result': 'nlp_sentiment_result',
            'summarize': 'summarize_text',
            'summary_result': 'nlp_summary_result',
            'text_plan': 'nlp_text_plan',
            'topic_assign_result': 'nlp_topic_assign_result',
            'topic_plan': 'nlp_topic_plan',
            'topic_result': 'nlp_topic_result',
        },
    },
    'online': {
        "mixin_key": 'online',
        "tier": 'domain',
        "warn_flat": True,
        "bindings": {
            'capability_matrix': 'online_capability_matrix',
            'eval_result': 'online_eval_result',
            'evaluate': 'evaluate_online',
            'fit': 'fit_online',
            'fit_result': 'online_fit_result',
            'load_bundle': 'load_online_bundle',
            'partial_fit': 'partial_fit_online',
            'plan': 'online_plan',
            'predict': 'predict_online',
            'predict_result': 'online_predict_result',
            'save_bundle': 'save_online_bundle',
            'update_result': 'online_update_result',
        },
    },
    'preprocess': {
        "mixin_key": 'preprocess',
        "tier": 'core',
        "warn_flat": False,
        "bindings": {
            'apply_custom_transform': 'apply_custom_transform',
            'apply_preprocess_plans': 'apply_preprocess_plans',
            'bin': 'bin',
            'binning_plan': 'binning_plan',
            'custom_plan': 'custom_plan',
            'date_plan': 'date_plan',
            'drop_columns': 'drop_columns',
            'encode': 'encode',
            'encode_plan': 'encode_plan',
            'extract_dates': 'extract_dates',
            'feature_select_plan': 'feature_select_plan',
            'handle_outliers': 'handle_outliers',
            'impute': 'impute',
            'impute_plan': 'impute_plan',
            'last_preprocess': 'last_preprocess',
            'list_transforms': 'list_transforms',
            'outlier_plan': 'outlier_plan',
            'reduce_dimensions': 'reduce_dimensions',
            'reduce_plan': 'reduce_plan',
            'register_transform': 'register_transform',
            'resample': 'resample',
            'resample_plan': 'resample_plan',
            'resample_strategies': 'resample_strategies',
            'scale': 'scale',
            'scale_plan': 'scale_plan',
            'select_features': 'select_features',
            'text_features': 'text_features',
            'text_plan': 'text_plan',
        },
    },
    'probabilistic': {
        "mixin_key": 'probabilistic',
        "tier": 'domain',
        "warn_flat": True,
        "bindings": {
            'capability_matrix': 'probabilistic_capability_matrix',
            'eval_result': 'probabilistic_eval_result',
            'evaluate': 'evaluate_probabilistic',
            'fit': 'fit_probabilistic',
            'fit_result': 'probabilistic_fit_result',
            'interval_result': 'probabilistic_interval_result',
            'load_bundle': 'load_probabilistic_bundle',
            'plan': 'probabilistic_plan',
            'predict': 'predict_probabilistic',
            'predict_interval': 'predict_interval',
            'predict_result': 'probabilistic_predict_result',
            'save_bundle': 'save_probabilistic_bundle',
        },
    },
    'rag': {
        "mixin_key": 'rag',
        "tier": 'domain',
        "warn_flat": True,
        "bindings": {
            'capability_matrix': 'rag_capability_matrix',
            'chunk': 'rag_chunk',
            'delete': 'rag_delete',
            'embed_and_index': 'rag_embed_and_index',
            'eval_result': 'rag_eval_result',
            'evaluate': 'rag_evaluate',
            'generate': 'rag_generate',
            'generate_result': 'rag_generate_result',
            'index_result': 'rag_index_result',
            'ingest_corpus': 'rag_ingest_corpus',
            'load_bundle': 'load_rag_bundle',
            'retrieve': 'rag_retrieve',
            'retrieve_result': 'rag_retrieve_result',
            'save_bundle': 'save_rag_bundle',
            'upsert': 'rag_upsert',
        },
    },
    'ranking': {
        "mixin_key": 'ranking',
        "tier": 'domain',
        "warn_flat": True,
        "bindings": {
            'capability_matrix': 'ranking_capability_matrix',
            'eval_result': 'ranker_eval_result',
            'evaluate': 'evaluate_ranker',
            'fit': 'fit_ranker',
            'fit_result': 'ranker_fit_result',
            'load_bundle': 'load_ranker_bundle',
            'plan': 'ranker_plan',
            'rank': 'rank',
            'rank_result': 'ranker_rank_result',
            'save_bundle': 'save_ranker_bundle',
        },
    },
    'recommender': {
        "mixin_key": 'recommender',
        "tier": 'domain',
        "warn_flat": True,
        "bindings": {
            'capability_matrix': 'recommender_capability_matrix',
            'eval_result': 'recommender_eval_result',
            'evaluate': 'evaluate_recommender',
            'fit': 'fit_recommender',
            'fit_result': 'recommender_fit_result',
            'load_bundle': 'load_recommender_bundle',
            'plan': 'recommender_plan',
            'recommend': 'recommend',
            'recommend_result': 'recommender_recommend_result',
            'save_bundle': 'save_recommender_bundle',
        },
    },
    'rl': {
        "mixin_key": 'rl',
        "tier": 'experimental',
        "warn_flat": True,
        "bindings": {
            'act': 'act_rl',
            'act_result': 'rl_act_result',
            'capability_matrix': 'rl_capability_matrix',
            'eval_result': 'rl_eval_result',
            'evaluate': 'evaluate_rl',
            'evaluate_imitation': 'evaluate_imitation',
            'fit': 'fit_rl',
            'fit_imitation': 'fit_imitation',
            'fit_result': 'rl_fit_result',
            'imitation_eval_result': 'imitation_eval_result',
            'imitation_fit_result': 'imitation_fit_result',
            'imitation_plan': 'imitation_plan',
            'imitation_predict_result': 'imitation_predict_result',
            'load_bundle': 'load_rl_bundle',
            'load_imitation_bundle': 'load_imitation_bundle',
            'plan': 'rl_plan',
            'predict_imitation': 'predict_imitation_action',
            'save_bundle': 'save_rl_bundle',
            'save_imitation_bundle': 'save_imitation_bundle',
        },
    },
    'semisupervised': {
        "mixin_key": 'semisupervised',
        "tier": 'domain',
        "warn_flat": True,
        "bindings": {
            'capability_matrix': 'semisupervised_capability_matrix',
            'eval_result': 'semisupervised_eval_result',
            'evaluate': 'evaluate_semisupervised',
            'fit': 'fit_semisupervised',
            'fit_result': 'semisupervised_fit_result',
            'load_bundle': 'load_semisupervised_bundle',
            'plan': 'semisupervised_plan',
            'predict': 'predict_semisupervised',
            'predict_result': 'semisupervised_predict_result',
            'save_bundle': 'save_semisupervised_bundle',
        },
    },
    'ssl': {
        "mixin_key": 'selfsupervised',
        "tier": 'domain',
        "warn_flat": True,
        "bindings": {
            'capability_matrix': 'ssl_capability_matrix',
            'eval_result': 'ssl_eval_result',
            'evaluate': 'evaluate_ssl',
            'finetune_head': 'finetune_ssl_head',
            'fit_pretext': 'fit_ssl_pretext',
            'fit_result': 'ssl_fit_result',
            'head_fit_result': 'ssl_head_fit_result',
            'head_plan': 'ssl_head_plan',
            'load_bundle': 'load_ssl_bundle',
            'plan': 'ssl_plan',
            'save_bundle': 'save_ssl_bundle',
            'transform': 'transform_ssl',
            'transform_result': 'ssl_transform_result',
        },
    },
    'symbolic': {
        "mixin_key": 'symbolic',
        "tier": 'domain',
        "warn_flat": True,
        "bindings": {
            'capability_matrix': 'symbolic_capability_matrix',
            'eval_result': 'symbolic_eval_result',
            'evaluate': 'evaluate_symbolic',
            'evaluate_neuro': 'evaluate_neuro_symbolic',
            'fit': 'fit_symbolic',
            'fit_neuro': 'fit_neuro_symbolic',
            'fit_result': 'symbolic_fit_result',
            'load_bundle': 'load_symbolic_bundle',
            'neuro_fit_result': 'neuro_symbolic_fit_result',
            'neuro_plan': 'neuro_symbolic_plan',
            'neuro_predict_result': 'neuro_symbolic_predict_result',
            'plan': 'symbolic_plan',
            'predict': 'predict_symbolic',
            'predict_neuro': 'predict_neuro_symbolic',
            'predict_result': 'symbolic_predict_result',
            'save_bundle': 'save_symbolic_bundle',
        },
    },
    'synthetic': {
        "mixin_key": 'synthetic',
        "tier": 'domain',
        "warn_flat": True,
        "bindings": {
            'capability_matrix': 'synthetic_capability_matrix',
            'eval_result': 'synthetic_eval_result',
            'evaluate': 'evaluate_synthetic',
            'fit': 'fit_synthesizer',
            'fit_result': 'synthetic_fit_result',
            'load_bundle': 'load_synthetic_bundle',
            'plan': 'synthesizer_plan',
            'sample': 'sample_synthetic',
            'sample_result': 'synthetic_sample_result',
            'save_bundle': 'save_synthetic_bundle',
        },
    },
    'tda': {
        "mixin_key": 'tda',
        "tier": 'experimental',
        "warn_flat": True,
        "bindings": {
            'capability_matrix': 'tda_capability_matrix',
            'eval_result': 'tda_eval_result',
            'evaluate': 'evaluate_tda',
            'fit': 'fit_tda',
            'fit_result': 'tda_fit_result',
            'load_bundle': 'load_tda_bundle',
            'plan': 'tda_plan',
            'predict': 'predict_tda',
            'predict_result': 'tda_predict_result',
            'save_bundle': 'save_tda_bundle',
            'transform': 'transform_tda',
            'transform_result': 'tda_transform_result',
        },
    },
    'timeseries': {
        "mixin_key": 'timeseries',
        "tier": 'domain',
        "warn_flat": True,
        "bindings": {
            'analysis_result': 'ts_analysis_result',
            'analyze': 'analyze_timeseries',
            'capability_matrix': 'timeseries_capability_matrix',
            'decompose': 'ts_decompose',
            'diagnostics': 'ts_diagnostics',
        },
    },
    'unsupervised': {
        "mixin_key": 'unsupervised',
        "tier": 'domain',
        "warn_flat": True,
        "bindings": {
            'assign': 'assign_clusters',
            'assign_result': 'cluster_assign_result',
            'capability_matrix': 'unsupervised_capability_matrix',
            'eval_result': 'cluster_eval_result',
            'evaluate': 'evaluate_clusters',
            'fit': 'fit_clusters',
            'fit_result': 'cluster_fit_result',
            'load_bundle': 'load_unsupervised_bundle',
            'plan': 'cluster_plan',
            'save_bundle': 'save_unsupervised_bundle',
        },
    },
}

_PROPERTY_FACADE_NAMES: Final[frozenset[str]] = frozenset(
    {
        "plan",
        "result",
        "transcript",
        "assumptions",
        "spec",
        "last_report",
        "history",
        "last_dry_run",
        "last_summary",
        "last_walkthrough",
        "backbone",
        "backbone_head",
        "asr_eval",
        "speech_result",
        "train_result",
        "cv_result",
        "search_result",
        "nested_cv_result",
        "export_result",
        "ddp_result",
        "text_plan",
        "topic_plan",
        "head_plan",
        "neuro_plan",
        "imitation_plan",
        "imitation_fit_result",
        "imitation_eval_result",
        "imitation_predict_result",
        "analysis_result",
    }
)


def preferred_path(flat_name: str) -> str | None:
    """Map a flat Session method name to its preferred facade path.

    Used by discovery helpers and deprecation warnings so teaching and runtime
    point at the same ``session.<domain>.<method>`` spelling.

    Parameters
    ----------
    flat_name:
        Public flat method such as ``evaluate_fairness`` or ``fit``.

    Returns
    -------
    str or None
        ``session.<domain>.<method>`` when registered, otherwise ``None``.
    """
    for attr, spec in DOMAIN_FACADES.items():
        for facade_name, flat in spec["bindings"].items():
            if flat == flat_name:
                return f"session.{attr}.{facade_name}"
    return None


def resolve_operation_name(flat_or_facade: str) -> str:
    """Normalize flat or facade-style names to the canonical flat Session method.

    Accepts ``evaluate_fairness``, ``fairness.evaluate``, and
    ``session.fairness.evaluate``. Unrecognized spellings are returned cleaned
    so callers keep their existing unknown-name errors. Catalog keys remain
    flat; this is the dual-form input boundary for explain / AI / discovery.

    Parameters
    ----------
    flat_or_facade:
        Flat Session method or ``domain.method`` / ``session.domain.method``.

    Returns
    -------
    str
        Canonical flat method name when a facade path is recognized; otherwise
        the stripped input.
    """
    cleaned = str(flat_or_facade or "").strip()
    if not cleaned:
        return cleaned
    path = cleaned
    if path.startswith("session."):
        path = path[len("session.") :]
    if "." not in path:
        return cleaned
    attr, method = path.split(".", 1)
    spec = DOMAIN_FACADES.get(attr)
    if spec is None or method not in spec["bindings"]:
        return cleaned
    return str(spec["bindings"][method])


def flat_to_facade() -> dict[str, dict[str, Any]]:
    """Build the reverse index from flat Session methods to facade metadata.

    One lookup table for tier, warn policy, and preferred path for every flat
    Session member covered by the facade registry.

    Returns
    -------
    dict[str, dict[str, Any]]
        Flat method name → facade attr/method/tier/warn metadata.
    """
    out: dict[str, dict[str, Any]] = {}
    for attr, spec in DOMAIN_FACADES.items():
        for facade_name, flat in spec["bindings"].items():
            out[flat] = {
                "facade_attr": attr,
                "facade_method": facade_name,
                "preferred_path": f"session.{attr}.{facade_name}",
                "tier": spec["tier"],
                "warn_flat": spec["warn_flat"],
                "mixin_key": spec["mixin_key"],
            }
    return out


def _is_property_like_facade(facade_name: str) -> bool:
    if facade_name in _PROPERTY_FACADE_NAMES:
        return True
    if facade_name.endswith("_result") or facade_name.endswith("_plan"):
        return True
    return False


DEPRECATED_FLAT_ACTIONS: Final[frozenset[str]] = frozenset(
    flat
    for attr, spec in DOMAIN_FACADES.items()
    if spec["warn_flat"]
    for facade_name, flat in spec["bindings"].items()
    if not _is_property_like_facade(facade_name)
)


__all__ = [
    "DEPRECATED_FLAT_ACTIONS",
    "DOMAIN_FACADES",
    "DomainFacadeSpec",
    "flat_to_facade",
    "preferred_path",
    "resolve_operation_name",
]
