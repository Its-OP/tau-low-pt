"""Network wrapper for TwoTierPreFilter (P6 of prefilter expressiveness sweep).

Builds a two-tier Stage-1 prefilter: coarse MLP + kNN on all tracks,
top-N selection, then a richer refine MLP + wider kNN on the top-N
subset. Output shape matches the single-tier ``TrackPreFilter`` so the
existing training / eval / cascade loading infra is unchanged.

Defaults target the P6 sweep configuration:
    * top_n = 600
    * coarse: hidden=128, k=16, rounds=2
    * refine: hidden=384, k=32, rounds=3
    * edge features ON (E2a baseline)

Every kwarg below is overridable from the training CLI via
``--two-tier-*`` flags that ``train_prefilter.py`` forwards here.
"""

from weaver.nn.model.TwoTierPreFilter import TwoTierPreFilter
from weaver.utils.logger import _logger


_UNUSED_KWARGS = (
    'pretrained_backbone_path', 'backbone_mode',
    'mask_ce_loss_weight', 'confidence_loss_weight',
    'no_object_weight', 'num_decoder_layers', 'num_queries',
    'focal_bce_weight', 'potential_loss_weight',
    'beta_loss_weight', 'clustering_dim',
    'per_track_loss_weight', 'refinement_loss_weight',
    'num_enrichment_layers',
    # Single-tier-only flags: the two-tier path ignores these to keep
    # the train_prefilter argparse surface flat.
    'num_neighbors', 'num_message_rounds', 'aggregation_mode',
    'feature_embed_mode', 'feature_embed_dim',
    'use_feature_gate', 'feature_gate_bottleneck',
    'use_film_head', 'film_context_dim',
    'use_soft_attention_aggregation', 'soft_attention_bottleneck',
    'use_xgb_stub_feature', 'logit_adjust_tau', 'listwise_temperature',
    'ranking_temperature_start', 'ranking_temperature_end',
    'denoising_sigma_start', 'denoising_sigma_end',
    'drw_warmup_fraction', 'drw_positive_weight',
)


def get_model(data_config, **kwargs):
    """Build a TwoTierPreFilter configured from training-script kwargs."""
    for unused_arg in _UNUSED_KWARGS:
        kwargs.pop(unused_arg, None)

    input_dim = len(data_config.input_dicts['pf_features'])

    configuration = dict(
        input_dim=input_dim,
        top_n=kwargs.pop('two_tier_top_n', 600),
        coarse_hidden_dim=kwargs.pop('two_tier_coarse_hidden_dim', 128),
        refine_hidden_dim=kwargs.pop('two_tier_refine_hidden_dim', 384),
        coarse_neighbors=kwargs.pop('two_tier_coarse_neighbors', 16),
        refine_neighbors=kwargs.pop('two_tier_refine_neighbors', 32),
        coarse_message_rounds=kwargs.pop('two_tier_coarse_rounds', 2),
        refine_message_rounds=kwargs.pop('two_tier_refine_rounds', 3),
        use_edge_features=kwargs.pop('use_edge_features', True),
        dropout=kwargs.pop('dropout', 0.1),
        composite_offset=kwargs.pop('two_tier_composite_offset', 1e6),
        ranking_num_samples=50,
        loss_type=kwargs.pop('loss_type', 'pairwise'),
    )
    configuration.update(**kwargs)
    _logger.info('TwoTierPreFilter config: %s', configuration)

    model = TwoTierPreFilter(**configuration)

    total_params = sum(p.numel() for p in model.parameters())
    coarse_params = sum(p.numel() for p in model.coarse.parameters())
    refine_params = sum(p.numel() for p in model.refine.parameters())
    _logger.info(
        'TwoTierPreFilter: %d params (coarse %d + refine %d)',
        total_params, coarse_params, refine_params,
    )

    model_info = {
        'input_names': list(data_config.input_names),
        'input_shapes': {
            key: ((1,) + shape[1:])
            for key, shape in data_config.input_shapes.items()
        },
        'output_names': ['loss'],
        'dynamic_axes': {
            **{
                key: {0: 'N', 2: 'n_' + key.split('_')[0]}
                for key in data_config.input_names
            },
            **{'loss': {0: 'N'}},
        },
    }
    return model, model_info
