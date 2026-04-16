"""Stage 3 reranker: per-couple scoring head for the post-ParT cascade.

Designed by `reports/triplet_reranking/triplet_research_plan_20260408.md`
direction A. Each couple drawn from the ParT top-50 (after the loose
``m(ij) <= m_tau`` Filter A) is encoded as a flat 51-dim feature vector by
``part/utils/couple_features.py``, and this module scores each couple
independently — no cross-couple interactions.

Architecture (per-couple, no graph between couples):

    input projection : Conv1d(51 → 256, kernel_size=1) + BN + ReLU
    encoder          : 4 × ResidualBlock(256, dropout=0.1)
    scoring head     : Conv1d(256 → 128) + BN + ReLU + Dropout
                       + Conv1d(128 → 1)

The Conv1d-with-kernel-size-1 idiom means each couple is processed
independently along the ``C_max`` axis — exactly the per-element scoring
shape used by the prefilter (``weaver/weaver/nn/model/TrackPreFilter.py``)
for tracks. Variable couple counts per event are handled by padding to
``C_max`` and masking padded positions in the loss.

Loss: pairwise ranking loss in the same form used by the prefilter and the
CascadeReranker (``L = T · softplus((s_neg − s_pos) / T)``), with N=50
random negatives sampled per positive in each event. Events with no GT
couples are skipped.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional


class NanSafeBatchNorm1d(nn.BatchNorm1d):
    """BatchNorm1d that skips running stat updates when inputs contain NaN/Inf.

    Standard BatchNorm1d uses an EMA to update running_mean and running_var:
        running_mean = (1 - momentum) * running_mean + momentum * batch_mean

    If even one element is NaN, batch_mean becomes NaN, and the running
    stats are permanently corrupted. This wrapper detects non-finite values
    and skips the stat update for that batch, while still producing finite
    output by replacing NaN/Inf with 0 before the BN forward pass.

    State dict is identical to nn.BatchNorm1d — fully compatible for
    loading/saving checkpoints.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return super().forward(x)

        has_non_finite = not torch.isfinite(x).all()
        if not has_non_finite:
            # Fast path: clean batch, normal BN forward
            return super().forward(x)

        # Slow path: replace non-finite values with 0, run BN without
        # updating running stats (eval-mode forward), then restore training.
        clean_x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        self.training = False
        output = super().forward(clean_x)
        self.training = True
        return output


class ResidualBlock(nn.Module):
    """Post-activation residual block: 2× Conv1d(1×1) + BN + skip + ReLU.

    Math:
        y = ReLU( BN_2( conv_2( ReLU( BN_1( conv_1(x) ) ) ) ) + x )

    Same shape in / out. The skip is an identity (no projection) because
    the channel dimension is preserved at every block. Dropout sits after
    the inner ReLU.
    """

    def __init__(self, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.conv_1 = nn.Conv1d(
            hidden_dim, hidden_dim, kernel_size=1, bias=False,
        )
        self.batchnorm_1 = NanSafeBatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.conv_2 = nn.Conv1d(
            hidden_dim, hidden_dim, kernel_size=1, bias=False,
        )
        self.batchnorm_2 = NanSafeBatchNorm1d(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv_1(x)
        out = self.batchnorm_1(out)
        out = functional.relu(out)
        out = self.dropout(out)
        out = self.conv_2(out)
        out = self.batchnorm_2(out)
        out = out + identity
        return functional.relu(out)


class CoupleReranker(nn.Module):
    """Per-couple Stage 3 reranker.

    Args:
        input_dim: per-couple feature vector size (default 51, matches
            ``part/utils/couple_features.COUPLE_FEATURE_DIM``).
        hidden_dim: width of the encoder and the first scoring layer
            (default 256).
        num_residual_blocks: number of ResidualBlock(hidden_dim) layers
            in the encoder (default 4 → 8 hidden Conv1d layers).
        dropout: dropout rate inside residual blocks and the scoring head
            (default 0.1).
        ranking_num_samples: negatives sampled per positive in the
            pairwise ranking loss (default 50, matches the prefilter and
            CascadeReranker).
        ranking_temperature: softplus temperature in the ranking loss
            (default 1.0).
    """

    def __init__(
        self,
        input_dim: int = 51,
        hidden_dim: int = 256,
        num_residual_blocks: int = 4,
        dropout: float = 0.1,
        ranking_num_samples: int = 50,
        ranking_temperature: float = 1.0,
        couple_loss: str = 'pairwise',
        label_smoothing: float = 0.0,
        hardneg_fraction: float = 0.0,
        hardneg_margin: float = 0.1,
    ):
        super().__init__()
        if couple_loss not in ('pairwise', 'softmax-ce'):
            raise ValueError(
                f"couple_loss must be 'pairwise' or 'softmax-ce', "
                f"got '{couple_loss}'",
            )
        if not 0.0 <= hardneg_fraction <= 1.0:
            raise ValueError(
                f'hardneg_fraction must be in [0, 1], got {hardneg_fraction}',
            )
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_residual_blocks = num_residual_blocks
        self.dropout_rate = dropout
        self.ranking_num_samples = ranking_num_samples
        self.ranking_temperature = ranking_temperature
        self.couple_loss = couple_loss
        self.label_smoothing = label_smoothing
        self.hardneg_fraction = hardneg_fraction
        self.hardneg_margin = hardneg_margin

        # Input projection: 51 → hidden_dim
        self.input_projection = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=1, bias=False),
            NanSafeBatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Encoder: stack of residual blocks at width hidden_dim
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim=hidden_dim, dropout=dropout)
            for _ in range(num_residual_blocks)
        ])

        # Scoring head: hidden_dim → hidden_dim/2 → 1
        intermediate_dim = hidden_dim // 2
        self.scorer = nn.Sequential(
            nn.Conv1d(hidden_dim, intermediate_dim, kernel_size=1, bias=False),
            NanSafeBatchNorm1d(intermediate_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(intermediate_dim, 1, kernel_size=1),
        )

    def forward(self, couple_features: torch.Tensor) -> torch.Tensor:
        """Score each couple independently.

        Args:
            couple_features: ``(B, input_dim, C)`` per-couple feature tensor,
                where C is the per-event couple count (padded to batch max).

        Returns:
            ``(B, C)`` per-couple scores.
        """
        x = self.input_projection(couple_features)
        for block in self.residual_blocks:
            x = block(x)
        scores = self.scorer(x)  # (B, 1, C)
        return scores.squeeze(1)  # (B, C)

    def compute_loss(
        self,
        couple_features: torch.Tensor,
        couple_labels: torch.Tensor,
        couple_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Couple ranking loss — dispatches to the configured branch.

        Two branches are supported (selected via ``self.couple_loss``):

        * ``'pairwise'`` (default): softplus pairwise ranking loss
          ``L = T · softplus((s_neg − s_pos) / T)`` averaged over all
          (positive, negative) pairs per event, then over events.
          Mirrors ``CascadeReranker._pairwise_ranking_loss``.

        * ``'softmax-ce'``: listwise softmax cross-entropy (ListMLE
          top-1). For each positive ``p_i`` in an event, the loss is
          ``L_i = −s_{p_i}/T + logsumexp([s_{p_i}, s_{negs}]/T)``,
          optionally smoothed by ``self.label_smoothing``.

        Args:
            couple_features: ``(B, input_dim, C)`` per-couple features.
            couple_labels: ``(B, C)`` binary labels (1 = GT couple).
            couple_mask: ``(B, C)`` validity mask (1 = real, 0 = padded).

        Returns:
            dict with ``'total_loss'``, ``'ranking_loss'``, ``'_scores'``.
        """
        scores = self.forward(couple_features)  # (B, C)
        if self.couple_loss == 'softmax-ce':
            ranking_loss = self._softmax_ce_loss(
                scores, couple_labels, couple_mask,
            )
        else:
            ranking_loss = self._pairwise_loss(
                scores, couple_labels, couple_mask,
            )
        return {
            'total_loss': ranking_loss,
            'ranking_loss': ranking_loss,
            '_scores': scores,
        }

    def _pairwise_loss(
        self,
        scores: torch.Tensor,
        couple_labels: torch.Tensor,
        couple_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = scores.shape[0]
        temperature = self.ranking_temperature
        event_losses: list[torch.Tensor] = []

        for event_index in range(batch_size):
            event_scores = scores[event_index]
            event_labels = couple_labels[event_index]
            event_valid = couple_mask[event_index] > 0.5

            positive_indices = (
                (event_labels > 0.5) & event_valid
            ).nonzero(as_tuple=True)[0]
            negative_indices = (
                (event_labels < 0.5) & event_valid
            ).nonzero(as_tuple=True)[0]

            if len(positive_indices) == 0 or len(negative_indices) == 0:
                continue

            sampled_negatives = self._sample_negative_indices(
                event_scores, positive_indices, negative_indices,
            )

            positive_scores = event_scores[positive_indices].unsqueeze(1)
            negative_scores = event_scores[sampled_negatives].unsqueeze(0)

            scaled_margin = (negative_scores - positive_scores) / temperature
            pairwise_loss = temperature * functional.softplus(scaled_margin)
            event_losses.append(pairwise_loss.mean())

        if not event_losses:
            return scores.sum() * 0.0
        return torch.stack(event_losses).mean()

    def _sample_negative_indices(
        self,
        event_scores: torch.Tensor,
        positive_indices: torch.Tensor,
        negative_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Sample ``ranking_num_samples`` negative couple indices.

        When ``hardneg_fraction == 0`` the sample is fully random (legacy
        behavior). When > 0, the first ``⌈N · frac⌉`` slots are filled
        from the top-scoring negatives that pass a positive-aware margin
        filter (``score < max(pos_scores) − margin``); this is the ANCE /
        NV-Retriever recipe and prevents false negatives from being
        selected as hard negatives. The remaining slots are filled with
        random negatives to preserve gradient diversity.

        Math:
            N     = min(ranking_num_samples, |negs|)
            M     = min(⌈N · hardneg_fraction⌉, |negs|)
            pool  = top-M negatives by score, filtered by
                    score < max(s_pos) − hardneg_margin
            fill  = N − |pool| random negatives
            out   = concat(pool, fill)
        """
        num_samples = min(self.ranking_num_samples, len(negative_indices))
        if self.hardneg_fraction <= 0.0:
            sample_indices = torch.randint(
                0, len(negative_indices), (num_samples,),
                device=event_scores.device,
            )
            return negative_indices[sample_indices]

        hardneg_count = min(
            int(round(num_samples * self.hardneg_fraction)),
            len(negative_indices),
        )
        if hardneg_count > 0:
            with torch.no_grad():
                negative_scores_detached = event_scores[
                    negative_indices
                ].detach()
                top_values, top_positions = torch.topk(
                    negative_scores_detached,
                    k=min(hardneg_count, len(negative_indices)),
                )
                threshold = (
                    event_scores[positive_indices].detach().max()
                    - self.hardneg_margin
                )
                keep_mask = top_values < threshold
                hard_positions = top_positions[keep_mask]
            hard_negatives = negative_indices[hard_positions]
        else:
            hard_negatives = negative_indices.new_empty((0,))

        remaining = num_samples - len(hard_negatives)
        if remaining > 0:
            random_positions = torch.randint(
                0, len(negative_indices), (remaining,),
                device=event_scores.device,
            )
            random_negatives = negative_indices[random_positions]
            sampled = torch.cat([hard_negatives, random_negatives], dim=0)
        else:
            sampled = hard_negatives[:num_samples]
        return sampled

    def _softmax_ce_loss(
        self,
        scores: torch.Tensor,
        couple_labels: torch.Tensor,
        couple_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Listwise softmax cross-entropy loss.

        For each positive ``p_i`` in an event, build the candidate pool
        ``{p_i} ∪ sample(negs, N)`` and compute

            L_i = −log p_i + ε · (−1/M · Σ_j log p_j − log p_i)
                = (1 − ε) · (−s_{p_i}/T + logsumexp(s_pool/T))
                  + ε · (−mean(s_pool)/T + logsumexp(s_pool/T))

        where ``M = len(pool)`` and ``ε = self.label_smoothing``. With
        ``ε = 0`` this reduces to the plain listwise softmax-CE (a.k.a.
        ListMLE top-1), which directly optimizes P(positive ranked first).
        """
        batch_size = scores.shape[0]
        temperature = self.ranking_temperature
        eps = self.label_smoothing
        event_losses: list[torch.Tensor] = []

        for event_index in range(batch_size):
            event_scores = scores[event_index]
            event_labels = couple_labels[event_index]
            event_valid = couple_mask[event_index] > 0.5

            positive_indices = (
                (event_labels > 0.5) & event_valid
            ).nonzero(as_tuple=True)[0]
            negative_indices = (
                (event_labels < 0.5) & event_valid
            ).nonzero(as_tuple=True)[0]

            if len(positive_indices) == 0 or len(negative_indices) == 0:
                continue

            sampled_negatives = self._sample_negative_indices(
                event_scores, positive_indices, negative_indices,
            )
            negative_scores = event_scores[sampled_negatives]

            positive_losses: list[torch.Tensor] = []
            for positive_index in positive_indices:
                positive_score = event_scores[positive_index]
                pool_scores = torch.cat(
                    [positive_score.unsqueeze(0), negative_scores],
                )
                scaled_pool = pool_scores / temperature
                log_normalizer = torch.logsumexp(scaled_pool, dim=0)

                nll = -positive_score / temperature + log_normalizer
                if eps > 0.0:
                    pool_size = pool_scores.shape[0]
                    uniform_nll = (
                        -scaled_pool.mean() + log_normalizer
                    )
                    smoothed = (
                        (1.0 - eps) * nll + eps * uniform_nll
                    )
                    positive_losses.append(smoothed)
                else:
                    positive_losses.append(nll)

            event_losses.append(torch.stack(positive_losses).mean())

        if not event_losses:
            return scores.sum() * 0.0
        return torch.stack(event_losses).mean()
