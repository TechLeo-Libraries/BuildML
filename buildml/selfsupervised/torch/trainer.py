"""Train tabular Torch SSL encoders (train partition only)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from buildml.core.errors import ValidationError
from buildml.dl.extras import require_torch
from buildml.selfsupervised.torch import augment, losses, models


@dataclass(slots=True)
class SSLTrainConfig:
    method: str
    epochs: int = 40
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    temperature: float = 0.5
    byol_momentum: float = 0.996
    vicreg_sim_coeff: float = 25.0
    vicreg_std_coeff: float = 25.0
    vicreg_cov_coeff: float = 1.0
    vae_beta: float = 1.0
    mask_ratio: float = 0.3
    noise_std: float = 0.1
    feature_dropout: float = 0.1
    scale_jitter: float = 0.05
    device: str = "cpu"
    random_state: int | None = 0


@dataclass(slots=True)
class SSLTrainResult:
    module: Any
    target_module: Any | None
    pretext_loss: float
    reconstruction_mae: float | None
    method: str
    latent_dim: int
    n_features: int
    config: SSLTrainConfig
    history: tuple[float, ...] = ()


def train_tabular_ssl(
    x: np.ndarray,
    *,
    method: str,
    latent_dim: int,
    hidden: tuple[int, ...],
    projector_hidden: tuple[int, ...] = (64,),
    projector_dim: int = 32,
    predictor_hidden: tuple[int, ...] = (64,),
    config: SSLTrainConfig | None = None,
) -> SSLTrainResult:
    """Fit a tabular SSL module on feature matrix ``x`` (train only)."""
    require_torch(feature="SSL training")
    x_arr = np.asarray(x, dtype=np.float32)
    if x_arr.ndim != 2 or x_arr.shape[0] < 4:
        raise ValidationError(
            f"Tabular SSL requires at least 4 train rows (got shape={x_arr.shape})."
        )
    n_features = int(x_arr.shape[1])
    cfg = config or SSLTrainConfig(method=method)
    cfg.method = method
    rng = np.random.default_rng(cfg.random_state)
    torch = require_torch(feature="SSL training")
    device = torch.device(cfg.device)
    tensor_x = torch.as_tensor(x_arr, device=device)

    if method == "simclr_tabular":
        module = models.build_simclr_module(
            n_features,
            hidden=hidden,
            latent_dim=latent_dim,
            projector_hidden=projector_hidden,
            projector_dim=projector_dim,
        ).to(device)
        target = None
        train_fn = _train_contrastive_simclr
    elif method == "byol_tabular":
        module, target = models.build_byol_module(
            n_features,
            hidden=hidden,
            latent_dim=latent_dim,
            projector_hidden=projector_hidden,
            projector_dim=projector_dim,
            predictor_hidden=predictor_hidden,
        )
        module = module.to(device)
        target = target.to(device)
        train_fn = _train_contrastive_byol
    elif method == "vicreg_tabular":
        module = models.build_vicreg_module(
            n_features,
            hidden=hidden,
            latent_dim=latent_dim,
            projector_hidden=projector_hidden,
            projector_dim=projector_dim,
        ).to(device)
        target = None
        train_fn = _train_contrastive_vicreg
    elif method == "mae_tabular":
        module = models.build_mae_module(
            n_features, hidden=hidden, latent_dim=latent_dim
        ).to(device)
        target = None
        train_fn = _train_mae
    elif method == "vae_tabular":
        module = models.build_vae(
            n_features, hidden=hidden, latent_dim=latent_dim
        ).to(device)
        target = None
        train_fn = _train_vae
    else:
        raise ValidationError(f"Unsupported tabular Torch SSL method {method!r}")

    history = train_fn(module, target, tensor_x, cfg, rng)
    pretext_loss = float(history[-1]) if history else float("nan")
    recon_mae = _reconstruction_mae(module, tensor_x, method, cfg, rng)
    return SSLTrainResult(
        module=module,
        target_module=target,
        pretext_loss=pretext_loss,
        reconstruction_mae=recon_mae,
        method=method,
        latent_dim=latent_dim,
        n_features=n_features,
        config=cfg,
        history=tuple(history),
    )


def _iter_batches(
    x: Any, batch_size: int, rng: np.random.Generator
) -> list[Any]:
    n = int(x.shape[0])
    indices = rng.permutation(n)
    batches: list[Any] = []
    for start in range(0, n, batch_size):
        idx = indices[start : start + batch_size]
        if len(idx) >= 2:
            batches.append(x[idx])
    return batches


def _train_contrastive_simclr(
    module: Any,
    target: Any | None,
    x: Any,
    cfg: SSLTrainConfig,
    rng: np.random.Generator,
) -> list[float]:
    del target
    torch = require_torch(feature="SSL SimCLR training")
    opt = torch.optim.AdamW(
        module.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    losses_hist: list[float] = []
    for _ in range(cfg.epochs):
        epoch_loss = 0.0
        n_batches = 0
        for batch in _iter_batches(x, cfg.batch_size, rng):
            v1, v2 = augment.augment_tabular_pair(
                batch,
                noise_std=cfg.noise_std,
                feature_dropout=cfg.feature_dropout,
                scale_jitter=cfg.scale_jitter,
                rng=rng,
            )
            _, p1 = module(v1)
            _, p2 = module(v2)
            loss = losses.nt_xent_loss(p1, p2, temperature=cfg.temperature)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            epoch_loss += float(loss.detach().cpu())
            n_batches += 1
        losses_hist.append(epoch_loss / max(n_batches, 1))
    return losses_hist


def _train_contrastive_byol(
    module: Any,
    target: Any | None,
    x: Any,
    cfg: SSLTrainConfig,
    rng: np.random.Generator,
) -> list[float]:
    if target is None:
        raise ValidationError("BYOL requires a target network.")
    torch = require_torch(feature="SSL BYOL training")
    opt = torch.optim.AdamW(
        module.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    losses_hist: list[float] = []
    for _ in range(cfg.epochs):
        epoch_loss = 0.0
        n_batches = 0
        for batch in _iter_batches(x, cfg.batch_size, rng):
            v1, v2 = augment.augment_tabular_pair(
                batch,
                noise_std=cfg.noise_std,
                feature_dropout=cfg.feature_dropout,
                scale_jitter=cfg.scale_jitter,
                rng=rng,
            )
            _, pred = module(v1)
            with torch.no_grad():
                _, tgt = target(v2)
            loss = losses.byol_loss(pred, tgt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.ema_update(target, module, momentum=cfg.byol_momentum)
            epoch_loss += float(loss.detach().cpu())
            n_batches += 1
        losses_hist.append(epoch_loss / max(n_batches, 1))
    return losses_hist


def _train_contrastive_vicreg(
    module: Any,
    target: Any | None,
    x: Any,
    cfg: SSLTrainConfig,
    rng: np.random.Generator,
) -> list[float]:
    del target
    torch = require_torch(feature="SSL VICReg training")
    opt = torch.optim.AdamW(
        module.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    losses_hist: list[float] = []
    for _ in range(cfg.epochs):
        epoch_loss = 0.0
        n_batches = 0
        for batch in _iter_batches(x, cfg.batch_size, rng):
            v1, v2 = augment.augment_tabular_pair(
                batch,
                noise_std=cfg.noise_std,
                feature_dropout=cfg.feature_dropout,
                scale_jitter=cfg.scale_jitter,
                rng=rng,
            )
            _, z1 = module(v1)
            _, z2 = module(v2)
            loss = losses.vicreg_loss(
                z1,
                z2,
                sim_coeff=cfg.vicreg_sim_coeff,
                std_coeff=cfg.vicreg_std_coeff,
                cov_coeff=cfg.vicreg_cov_coeff,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            epoch_loss += float(loss.detach().cpu())
            n_batches += 1
        losses_hist.append(epoch_loss / max(n_batches, 1))
    return losses_hist


def _train_mae(
    module: Any,
    target: Any | None,
    x: Any,
    cfg: SSLTrainConfig,
    rng: np.random.Generator,
) -> list[float]:
    del target
    torch = require_torch(feature="SSL MAE training")
    opt = torch.optim.AdamW(
        module.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    losses_hist: list[float] = []
    for _ in range(cfg.epochs):
        epoch_loss = 0.0
        n_batches = 0
        for batch in _iter_batches(x, cfg.batch_size, rng):
            masked, mask = augment.random_feature_mask(
                batch, mask_ratio=cfg.mask_ratio, rng=rng
            )
            _, recon = module(masked)
            loss = losses.mae_reconstruction_loss(recon, batch, mask)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            epoch_loss += float(loss.detach().cpu())
            n_batches += 1
        losses_hist.append(epoch_loss / max(n_batches, 1))
    return losses_hist


def _train_vae(
    module: Any,
    target: Any | None,
    x: Any,
    cfg: SSLTrainConfig,
    rng: np.random.Generator,
) -> list[float]:
    del target
    torch = require_torch(feature="SSL VAE training")
    opt = torch.optim.AdamW(
        module.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    module.train()
    losses_hist: list[float] = []
    for _ in range(cfg.epochs):
        epoch_loss = 0.0
        n_batches = 0
        for batch in _iter_batches(x, cfg.batch_size, rng):
            recon, mu, logvar, _z = module(batch)
            loss = losses.vae_elbo_loss(
                recon, batch, mu, logvar, beta=cfg.vae_beta
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            epoch_loss += float(loss.detach().cpu())
            n_batches += 1
        losses_hist.append(epoch_loss / max(n_batches, 1))
    return losses_hist


def _reconstruction_mae(
    module: Any,
    x: Any,
    method: str,
    cfg: SSLTrainConfig,
    rng: np.random.Generator,
) -> float | None:
    torch = require_torch(feature="SSL diagnostics")
    module.eval()
    with torch.no_grad():
        if method == "mae_tabular":
            masked, mask = augment.random_feature_mask(
                x[: min(256, x.shape[0])],
                mask_ratio=cfg.mask_ratio,
                rng=rng,
            )
            _, recon = module(masked)
            diff = (recon - x[: recon.shape[0]]).abs()
            return float((diff * mask.float()).sum() / mask.float().sum().clamp(min=1.0))
        if method == "vae_tabular":
            recon, _mu, _logvar, _z = module(x[: min(256, x.shape[0])])
            return float((recon - x[: recon.shape[0]]).abs().mean().cpu())
    return None
