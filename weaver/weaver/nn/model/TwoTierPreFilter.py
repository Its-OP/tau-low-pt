"""Two-tier Stage-1 prefilter (P6 of the prefilter expressiveness sweep).

Stacks two ``TrackPreFilter`` instances inside Stage 1:

1. **Coarse** — lightweight per-track MLP + kNN message passing on all
   tracks. Outputs a score per track; selects the top-N candidates.
2. **Refine** — richer per-track MLP + wider kNN on the top-N subset
   only. Re-ranks them.

Composite output keeps the ``(B, P)`` shape of a single-tier prefilter:

    composite[b, i] = refine_score  if track i is in coarse-top-N
                    = coarse_score − offset  otherwise (pushed well
                                                        below the refine range)
    composite[b, i] = −∞            if mask[b, 0, i] == 0 (padded)

The offset construction guarantees non-selected valid tracks can never
rank above any selected track, so the eventual top-K1 selection always
draws from the refine-ranked subset. Positives that the coarse stage
drops below the top-N cut contribute a very large pairwise-ranking loss
(their composite score is ``-offset``, negatives typically score around
0), which pressures the coarse stage to pull them in.

Training ties both tiers together through one ranking loss on the
composite scores — the refine stage learns the fine ordering and the
coarse stage learns the coarse filtering, end-to-end under the same
loss family used by single-tier ``TrackPreFilter``.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from weaver.nn.model.TrackPreFilter import TrackPreFilter


class TwoTierPreFilter(nn.Module):
    """Coarse-then-refine Stage-1 prefilter.

    The coarse and refine tiers are full :class:`TrackPreFilter`
    instances configured independently. The top-N cut between them is
    a hard top-K by coarse score; downstream gradients flow into the
    coarse tier through the composite-loss path, not through the
    non-differentiable top-K argument selection itself.

    Args:
        input_dim: raw feature count (``16`` for the current pipeline).
        top_n: number of tracks the refine tier sees (must exceed the
            downstream ``K1=256`` cascade operating point).
        coarse_hidden_dim / refine_hidden_dim: hidden widths.
        coarse_neighbors / refine_neighbors: kNN k for each tier.
        coarse_message_rounds / refine_message_rounds: MP round counts.
        use_edge_features: whether both tiers use pairwise-LV edge
            features (defaults to ``True`` to match E2a).
        dropout: shared dropout rate for the MLP + scorer stacks.
        composite_offset: magnitude of the score shift applied to
            non-selected valid tracks so they always rank below
            selected ones. ``1e6`` is overkill-safe for any ranking
            loss scale.
        ranking_num_samples: passed to both tiers' ranking loss.
        loss_type: passed to both tiers (``'pairwise'`` is the default).
        extra_prefilter_kwargs: additional TrackPreFilter kwargs
            applied to BOTH tiers.
    """

    def __init__(
        self,
        input_dim: int = 16,
        top_n: int = 600,
        coarse_hidden_dim: int = 128,
        refine_hidden_dim: int = 384,
        coarse_neighbors: int = 16,
        refine_neighbors: int = 32,
        coarse_message_rounds: int = 2,
        refine_message_rounds: int = 3,
        use_edge_features: bool = True,
        dropout: float = 0.0,
        composite_offset: float = 1e6,
        ranking_num_samples: int = 50,
        loss_type: str = 'pairwise',
        extra_prefilter_kwargs: dict | None = None,
    ):
        super().__init__()
        if top_n <= 0:
            raise ValueError(f'top_n must be positive, got {top_n}')
        if refine_neighbors >= top_n:
            raise ValueError(
                f'refine_neighbors ({refine_neighbors}) must be < top_n '
                f'({top_n}); the refine tier runs kNN over the top-N '
                f'subset and cannot ask for more neighbours than it sees.',
            )
        self.top_n = top_n
        self.composite_offset = float(composite_offset)
        self.loss_type = loss_type
        self.ranking_num_samples = ranking_num_samples

        shared = dict(
            mode='mlp',
            input_dim=input_dim,
            use_edge_features=use_edge_features,
            dropout=dropout,
            loss_type=loss_type,
            ranking_num_samples=ranking_num_samples,
        )
        if extra_prefilter_kwargs:
            shared.update(extra_prefilter_kwargs)

        self.coarse = TrackPreFilter(
            hidden_dim=coarse_hidden_dim,
            num_neighbors=coarse_neighbors,
            num_message_rounds=coarse_message_rounds,
            **shared,
        )
        self.refine = TrackPreFilter(
            hidden_dim=refine_hidden_dim,
            num_neighbors=refine_neighbors,
            num_message_rounds=refine_message_rounds,
            **shared,
        )

    # --- Forward / loss ----------------------------------------------------

    def forward(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        lorentz_vectors: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return the composite per-track scores ``(B, P)``."""
        valid_mask = mask.squeeze(1).bool()

        # Tier 1: coarse pass over all tracks.
        coarse_scores = self.coarse(points, features, lorentz_vectors, mask)

        # Top-N selection by coarse score (padded tracks excluded).
        coarse_for_topk = coarse_scores.masked_fill(
            ~valid_mask, float('-inf'),
        )
        batch_size, num_positions = coarse_scores.shape
        effective_top_n = min(self.top_n, num_positions)
        top_n_indices = coarse_for_topk.topk(
            effective_top_n, dim=1,
        ).indices  # (B, N)

        # Gather top-N inputs. Shapes: features (B, F, P), points (B, 2, P),
        # lorentz (B, 4, P), mask (B, 1, P).
        index_broadcast = top_n_indices.unsqueeze(1)  # (B, 1, N)
        features_sel = torch.gather(
            features, 2,
            index_broadcast.expand(-1, features.shape[1], -1),
        )
        points_sel = torch.gather(
            points, 2,
            index_broadcast.expand(-1, points.shape[1], -1),
        )
        lorentz_sel = torch.gather(
            lorentz_vectors, 2,
            index_broadcast.expand(-1, lorentz_vectors.shape[1], -1),
        )
        mask_sel = torch.gather(mask, 2, index_broadcast)  # (B, 1, N)

        # Tier 2: refine on the selected subset.
        refine_scores_sel = self.refine(
            points_sel, features_sel, lorentz_sel, mask_sel,
        )  # (B, N)

        # Composite scores. Non-selected valid tracks fall back to
        # ``coarse − offset`` so they are guaranteed below every
        # selected track. Selected tracks (including accidental
        # padded positions when a batch has fewer valid tracks than
        # ``top_n``) are overwritten by refine scores, and the final
        # ``masked_fill`` zeros out any -inf paddings.
        composite = coarse_scores - self.composite_offset
        composite = composite.scatter(1, top_n_indices, refine_scores_sel)
        composite = composite.masked_fill(~valid_mask, float('-inf'))
        return composite

    def compute_loss(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        lorentz_vectors: torch.Tensor,
        mask: torch.Tensor,
        track_labels: torch.Tensor,
        use_contrastive_denoising: bool = True,
    ) -> dict[str, torch.Tensor]:
        """One ranking loss on the composite scores — trains both tiers
        through the same gradient path.

        The coarse tier's weights receive gradient through ``composite =
        coarse − offset`` (non-selected positions) and through the
        scatter-target mask (``top_n_indices`` itself is non-
        differentiable, but the selected slice's coarse-stage grads
        flow via the refine sub-graph). The refine tier gets gradient
        from the refine scores that land inside the composite for
        selected tracks.

        ``use_contrastive_denoising`` is accepted for API parity with
        :class:`TrackPreFilter.compute_loss` but not yet applied —
        denoising on the two-tier path needs a joint-tier formulation
        that is out of scope for this smoke-ready implementation.
        """
        del use_contrastive_denoising  # see docstring.
        scores = self.forward(points, features, lorentz_vectors, mask)
        valid_mask = mask.squeeze(1).bool()
        labels_flat = (
            track_labels.squeeze(1)[:, :scores.shape[1]] * valid_mask.float()
        )
        ranking_loss = self.coarse._ranking_loss(
            scores, labels_flat, valid_mask,
        )
        return {
            'ranking_loss': ranking_loss,
            'total_loss': ranking_loss,
            # Raw composite scores — popped by train_prefilter's
            # ``validate`` path to feed the MetricsAccumulator.
            '_scores': scores,
        }

    # --- Schedule pass-throughs -------------------------------------------

    def set_temperature_progress(self, progress: float) -> None:
        self.coarse.set_temperature_progress(progress)
        self.refine.set_temperature_progress(progress)

    def set_drw_active(self, active: bool) -> None:
        self.coarse.set_drw_active(active)
        self.refine.set_drw_active(active)

    @property
    def current_ranking_temperature(self) -> float:
        return self.coarse.current_ranking_temperature

    @property
    def current_denoising_sigma(self) -> float:
        return self.coarse.current_denoising_sigma

    @property
    def drw_warmup_fraction(self) -> float:
        return self.coarse.drw_warmup_fraction
