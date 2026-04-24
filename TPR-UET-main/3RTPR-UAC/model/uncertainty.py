"""
Uncertainty-Aware Module for 3RTPR-UAC.

This module implements three types of uncertainty estimation for text-based person retrieval:

1. EPISTEMIC UNCERTAINTY (model uncertainty):
   - Estimated via MC Dropout at inference time
   - Multiple forward passes with dropout enabled
   - Variance of predictions = epistemic uncertainty
   - High epistemic uncertainty → model doesn't know the answer (should learn more)

2. ALEATORIC UNCERTAINTY (data uncertainty):
   - Estimated from the similarity score distribution within a batch
   - Entropy of cross-modal matching scores
   - Label noise from noisy correspondences
   - High aleatoric uncertainty → inherent ambiguity in data (irreducible)

3. COMBINED UNCERTAINTY:
   - Aleatoric + Epistemic for calibrated sample weighting
   - Used in uncertainty-aware CCD (UACS-2)
   - Used in uncertainty-calibrated SDM loss

References:
- Kendall & Gal, "What Uncertainties Do We Need in Bayesian Deep Learning?", 2017
- Used for text-based person retrieval with noisy correspondences
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AleatoricUncertaintyEstimator(nn.Module):
    """
    Estimates aleatoric uncertainty from cross-modal similarity distributions.

    Aleatoric uncertainty captures:
    - Ambiguity in text-image matching (multiple similar persons)
    - Label noise in training pairs (noisy correspondences)
    - Intra-class variability (same person, different appearance)

    Estimation method:
    - Compute entropy of matching score distribution per sample
    - High entropy = high uncertainty (data is ambiguous)
    """

    def __init__(self, temperature=0.02):
        super().__init__()
        self.temperature = temperature

    def forward(self, sim_matrix, pids):
        """
        Args:
            sim_matrix: [B, B] cross-modal similarity matrix (T→I)
            pids: [B] person IDs for the batch

        Returns:
            aleatoric_uncertainty: [B] per-sample aleatoric uncertainty
            per_sample_entropy: [B] entropy of each sample's matching distribution
        """
        B = sim_matrix.size(0)

        # Softmax distribution over similarity scores
        sim_probs = F.softmax(sim_matrix / self.temperature, dim=1)

        # Entropy: H(p) = -sum(p * log(p))
        # High entropy = high uncertainty
        epsilon = 1e-10
        entropy = -torch.sum(sim_probs * torch.log(sim_probs + epsilon), dim=1)

        # Normalize entropy to [0, 1] range
        # Max entropy for B classes = log(B)
        max_entropy = math.log(B + epsilon)
        normalized_entropy = entropy / max_entropy

        # Cross-modal agreement: does the model agree across directions?
        sim_T2I = sim_matrix
        sim_I2T = sim_matrix.t()

        # Agreement score: how consistent are T→I and I→T rankings?
        rank_agreement = []
        for i in range(B):
            rank_T2I = torch.argsort(sim_T2I[i], descending=True)
            rank_I2T = torch.argsort(sim_I2T[i], descending=True)
            # Count agreement in top-K
            k = min(10, B)
            topk_T2I = rank_T2I[:k]
            topk_I2T = rank_I2T[:k]
            agreement = len(set(topk_T2I.tolist()) & set(topk_I2T.tolist())) / k
            rank_agreement.append(agreement)

        rank_agreement = torch.tensor(rank_agreement, device=sim_matrix.device)

        # Aleatoric uncertainty = inverse of agreement + normalized entropy
        # High entropy + low agreement = high aleatoric uncertainty
        uncertainty = (1 - rank_agreement) * 0.5 + normalized_entropy * 0.5

        return uncertainty, normalized_entropy


class EpistemicUncertaintyEstimator(nn.Module):
    """
    Estimates epistemic uncertainty via MC Dropout.

    Epistemic uncertainty captures:
    - Model's confidence in its predictions
    - Regions of feature space not well-represented in training
    - Can be reduced with more data

    Estimation method:
    - Run multiple forward passes with dropout enabled
    - Compute variance of the predictions
    - High variance = model doesn't know (should learn more)
    """

    def __init__(self, n_mc_samples=5):
        super().__init__()
        self.n_mc_samples = n_mc_samples

    def forward(self, model, batch, embed_cache=None):
        """
        Args:
            model: DATPS model with dropout enabled
            batch: input batch dict
            embed_cache: pre-computed embeddings to speed up (optional)

        Returns:
            epistemic_uncertainty: [B] per-sample epistemic uncertainty
            mc_logits: [n_mc_samples, B, B] MC samples of similarity matrices
        """
        device = next(model.parameters()).device
        B = batch['images_a'].size(0)

        model.train()  # Ensure dropout is active

        mc_image_feats = []
        mc_text_feats = []

        for _ in range(self.n_mc_samples):
            # Forward pass through model
            output = model(batch)

            img_feat = output["image_norms_fused_feats"]
            txt_feat = output["text_norms_fused_feats"]

            mc_image_feats.append(img_feat)
            mc_text_feats.append(txt_feat)

        mc_image_feats = torch.stack(mc_image_feats, dim=0)  # [n_mc, B, D]
        mc_text_feats = torch.stack(mc_text_feats, dim=0)   # [n_mc, B, D]

        # Compute similarity matrices for each MC sample
        mc_logits = torch.einsum('nid,njd->nij', mc_text_feats, mc_image_feats)

        # Mean prediction
        mean_logits = mc_logits.mean(dim=0)  # [B, B]

        # Variance of predictions (epistemic uncertainty)
        logits_variance = mc_logits.var(dim=0)  # [B, B]
        logits_std = mc_logits.std(dim=0)       # [B, B]

        # Per-sample uncertainty: variance of diagonal elements
        # (matching pairs are on diagonal)
        diag_variance = logits_variance.diagonal()  # [B]

        # Normalize by the number of MC samples (Var of mean estimate)
        # Standard error of the mean
        per_sample_uncertainty = diag_variance / self.n_mc_samples

        # Also compute uncertainty from off-diagonal variance (negatives)
        off_diag_mask = ~torch.eye(B, dtype=torch.bool, device=device)
        off_diag_variance = logits_variance[off_diag_mask].view(B, B - 1).mean(dim=1)

        # Final epistemic uncertainty: combination of diagonal and off-diagonal
        epistemic = (diag_variance + off_diag_variance) / 2
        epistemic = epistemic / (epistemic.max() + 1e-8)  # Normalize to [0, 1]

        model.eval()  # Back to eval mode

        return epistemic, mc_logits


class UncertaintyCalibrator(nn.Module):
    """
    Combines aleatoric and epistemic uncertainty for calibrated sample weighting.

    The combined uncertainty is used for:
    1. Uncertainty-aware loss weighting (reduce loss for uncertain samples)
    2. Uncertainty-aware CCD (UACS-2) for better noise filtering
    3. Dynamic margin adjustment based on uncertainty

    Calibration formula (inspired by Kendall & Gal, 2017):
    Total uncertainty ≈ Aleatoric + Epistemic
    For regression: loss = (1/σ²) * MSE + log(σ)
    For classification: uncertainty-aware weighting

    In our case:
    - High epistemic: model uncertain → should learn more (lower weight in loss if noisy)
    - High aleatoric: data ambiguous → irreducible, use as regularization
    - Combined: use for adaptive confidence weighting
    """

    def __init__(self, epistemic_weight=0.5, aleatoric_weight=0.5):
        super().__init__()
        self.epistemic_weight = epistemic_weight
        self.aleatoric_weight = aleatoric_weight

    def forward(self, epistemic_unc, aleatoric_unc):
        """
        Args:
            epistemic_unc: [B] epistemic uncertainty per sample
            aleatoric_unc: [B] aleatoric uncertainty per sample

        Returns:
            combined_uncertainty: [B] calibrated combined uncertainty
            uncertainty_weights: [B] inverse uncertainty weights for loss
        """
        # Combine uncertainties
        combined = (self.epistemic_weight * epistemic_unc +
                    self.aleatoric_weight * aleatoric_unc)

        # Convert uncertainty to confidence weights
        # uncertainty_weight = 1 / (1 + uncertainty)
        # Range: [0.5, 1.0] — samples with high uncertainty get lower weight
        uncertainty_weights = 1.0 / (1.0 + combined)

        return combined, uncertainty_weights


class LogitsUncertaintyModule(nn.Module):
    """
    Per-dimension uncertainty estimation from logits.

    Assumes each logit dimension has independent noise.
    Learns per-dimension noise variance (aleatoric) or
    computes prediction variance (epistemic).

    This is useful for multi-scale uncertainty estimation
    when computing similarity at different granularity levels.
    """

    def __init__(self, input_dim, learn_aleatoric=True):
        super().__init__()
        self.learn_aleatoric = learn_aleatoric

        if learn_aleatoric:
            # Learn per-sample, per-dimension log variance
            self.log_noise_scale = nn.Parameter(torch.zeros(1))

    def forward(self, logits, reduction='mean'):
        """
        Args:
            logits: [B, D] or [B, B] similarity/logit matrix
            reduction: 'mean', 'sum', or 'none'

        Returns:
            nll: negative log-likelihood
            total_uncertainty: estimated uncertainty
        """
        if self.learn_aleatoric:
            # Heteroscedastic uncertainty: learn noise scale
            noise_scale = torch.exp(self.log_noise_scale) + 1e-6
            uncertainty = noise_scale

            # NLL for Gaussian: 0.5 * (log(2πσ²) + x²/σ²)
            nll = 0.5 * (torch.log(2 * math.pi * noise_scale) +
                         logits ** 2 / noise_scale)

            if reduction == 'mean':
                nll = nll.mean()
            elif reduction == 'sum':
                nll = nll.sum()

            return nll, uncertainty

        return None, torch.zeros_like(logits).mean()


def compute_batch_uncertainty(sim_matrix, pids, temperature=0.02):
    """
    Convenience function to compute batch-level uncertainty metrics.

    Args:
        sim_matrix: [B, B] similarity matrix
        pids: [B] person IDs
        temperature: softmax temperature

    Returns:
        dict with uncertainty metrics for monitoring
    """
    B = sim_matrix.size(0)

    # Matching confidence: diagonal similarity
    diag_sim = sim_matrix.diagonal()

    # Cross-modal consistency: compare T→I and I→T
    sim_T2I = sim_matrix
    sim_I2T = sim_matrix.t()

    # Rank correlation between directions
    rank_corr = []
    for i in range(B):
        rank_T2I = torch.argsort(sim_T2I[i], descending=True)
        rank_I2T = torch.argsort(sim_I2T[i], descending=True)
        # Spearman-like correlation
        corr = 1 - 2 * torch.sum(torch.abs(rank_T2I.float() - rank_I2T.float())) / (B * (B - 1))
        rank_corr.append(corr)
    rank_corr = torch.stack(rank_corr)

    # Positive pair confidence vs negative pair confidence
    pos_mask = pids.unsqueeze(0) == pids.unsqueeze(1)
    neg_mask = ~pos_mask

    pos_sim = sim_T2I[pos_mask]
    neg_sim = sim_T2I[neg_mask]

    pos_mean = pos_sim.mean() if pos_sim.numel() > 0 else torch.tensor(0.0)
    neg_mean = neg_sim.mean() if neg_sim.numel() > 0 else torch.tensor(0.0)

    # Margin: how well separated are positive and negative pairs
    margin = pos_mean - neg_mean

    # Match uncertainty: how confident is the model about top-1 match
    top1_sim = sim_T2I.max(dim=1)[0]
    top1_uncertainty = 1 - top1_sim  # High when top match is low

    metrics = {
        'pos_sim_mean': pos_mean.item() if torch.is_tensor(pos_mean) else pos_mean,
        'neg_sim_mean': neg_mean.item() if torch.is_tensor(neg_mean) else neg_mean,
        'margin': margin.item() if torch.is_tensor(margin) else margin,
        'rank_correlation': rank_corr.mean().item(),
        'top1_uncertainty_mean': top1_uncertainty.mean().item(),
        'sim_matrix_std': sim_T2I.std().item(),
        'sim_matrix_entropy': (-F.softmax(sim_T2I / temperature, dim=1) *
                                F.log_softmax(sim_T2I / temperature, dim=1)).mean().item(),
    }

    return metrics
