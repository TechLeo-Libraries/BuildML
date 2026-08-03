"""Contrastive and generative SSL losses (tabular Torch)."""

from __future__ import annotations

from typing import Any

from buildml.dl.extras import require_torch


def nt_xent_loss(z_i: Any, z_j: Any, *, temperature: float = 0.5) -> Any:
    """Compute SimCLR NT-Xent loss on L2-normalized projector outputs.

    Treats augmented views from the same row as positives and all other batch
    pairs as negatives in a symmetric cross-entropy objective.

    Parameters
    ----------
    z_i, z_j:
        Projector outputs for two augmented views with shape ``(batch, dim)``.
    temperature:
        Softmax temperature scaling similarity logits.

    Returns
    -------
    torch.Tensor
        Scalar NT-Xent loss averaged over the batch.
    """
    torch = require_torch(feature="SSL SimCLR loss")
    z_i = torch.nn.functional.normalize(z_i, dim=1)
    z_j = torch.nn.functional.normalize(z_j, dim=1)
    n = z_i.shape[0]
    z = torch.cat([z_i, z_j], dim=0)
    sim = torch.mm(z, z.t()) / float(temperature)
    mask = torch.eye(2 * n, device=sim.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, float("-inf"))
    targets = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(sim.device)
    return torch.nn.functional.cross_entropy(sim, targets)


def byol_loss(pred: Any, target: Any) -> Any:
    """Compute BYOL loss as negative cosine similarity between views.

    Both predictor and target projector outputs are L2-normalized before the
    symmetric MSE-style BYOL objective is applied.

    Parameters
    ----------
    pred:
        Online network predictor output.
    target:
        Target network projector output (stop-gradient).

    Returns
    -------
    torch.Tensor
        Scalar BYOL loss averaged over the batch.
    """
    torch = require_torch(feature="SSL BYOL loss")
    pred = torch.nn.functional.normalize(pred, dim=1)
    target = torch.nn.functional.normalize(target, dim=1)
    return 2.0 - 2.0 * (pred * target).sum(dim=1).mean()


def vicreg_loss(
    z_i: Any,
    z_j: Any,
    *,
    sim_coeff: float = 25.0,
    std_coeff: float = 25.0,
    cov_coeff: float = 1.0,
) -> Any:
    """Compute VICReg invariance, variance, and covariance regularization.

    Encourages representation invariance across views while preventing collapse
    via per-dimension variance and off-diagonal covariance penalties.

    Parameters
    ----------
    z_i, z_j:
        Encoder outputs for two augmented views.
    sim_coeff:
        Weight on the invariance (MSE) term.
    std_coeff:
        Weight on the variance hinge penalty.
    cov_coeff:
        Weight on the covariance off-diagonal penalty.

    Returns
    -------
    torch.Tensor
        Weighted sum of invariance, variance, and covariance losses.
    """
    torch = require_torch(feature="SSL VICReg loss")
    repr_loss = torch.nn.functional.mse_loss(z_i, z_j)
    std_i = torch.sqrt(z_i.var(dim=0) + 1e-4)
    std_j = torch.sqrt(z_j.var(dim=0) + 1e-4)
    std_loss = torch.mean(torch.relu(1.0 - std_i)) + torch.mean(torch.relu(1.0 - std_j))
    z_i_c = z_i - z_i.mean(dim=0)
    z_j_c = z_j - z_j.mean(dim=0)
    cov_i = (z_i_c.T @ z_i_c) / max(z_i.shape[0] - 1, 1)
    cov_j = (z_j_c.T @ z_j_c) / max(z_j.shape[0] - 1, 1)
    off_diag_i = cov_i.pow(2).sum() - cov_i.diagonal().pow(2).sum()
    off_diag_j = cov_j.pow(2).sum() - cov_j.diagonal().pow(2).sum()
    cov_loss = off_diag_i + off_diag_j
    return (
        sim_coeff * repr_loss + std_coeff * std_loss + cov_coeff * cov_loss
    )


def mae_reconstruction_loss(
    recon: Any,
    target: Any,
    mask: Any,
) -> Any:
    """Compute masked MSE reconstruction loss for MAE tabular training.

    Averages squared error only over masked feature dimensions so the decoder
    is penalised on hidden inputs.

    Parameters
    ----------
    recon:
        Decoder reconstruction with the same shape as ``target``.
    target:
        Original unmasked batch used as reconstruction target.
    mask:
        Boolean mask marking masked feature positions.

    Returns
    -------
    torch.Tensor
        Scalar masked mean squared error.
    """
    torch = require_torch(feature="SSL MAE loss")
    diff = (recon - target) ** 2
    masked = diff * mask.float()
    denom = mask.float().sum().clamp(min=1.0)
    return masked.sum() / denom


def vae_elbo_loss(
    recon: Any,
    target: Any,
    mu: Any,
    logvar: Any,
    *,
    beta: float = 1.0,
) -> Any:
    """Compute Gaussian VAE reconstruction plus KL divergence loss.

    Combines mean squared reconstruction error with the standard closed-form
    KL term between the latent posterior and a standard normal prior.

    Parameters
    ----------
    recon:
        Decoder reconstruction of the input batch.
    target:
        Original input batch.
    mu, logvar:
        Latent Gaussian parameters from the encoder.
    beta:
        KL weight (beta-VAE coefficient).

    Returns
    -------
    torch.Tensor
        Scalar ELBO loss (reconstruction + weighted KL).
    """
    torch = require_torch(feature="SSL VAE loss")
    recon_loss = torch.nn.functional.mse_loss(recon, target, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl


def ema_update(target: Any, online: Any, momentum: float = 0.996) -> None:
    """Apply exponential moving average from online to target network.

    Used by BYOL to slowly track the target encoder/projector weights without
    backprop through the target branch.

    Parameters
    ----------
    target:
        Target network whose parameters are updated in place.
    online:
        Online network providing the source parameters.
    momentum:
        EMA decay factor in ``(0, 1)``.
    """
    torch = require_torch(feature="SSL BYOL EMA")
    with torch.no_grad():
        for t_param, o_param in zip(target.parameters(), online.parameters(), strict=False):
            t_param.data.mul_(momentum).add_(o_param.data, alpha=1.0 - momentum)
