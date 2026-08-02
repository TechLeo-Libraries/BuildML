"""Torch modules for tabular self-supervised learning."""

from __future__ import annotations

from typing import Any

from buildml.dl.extras import require_torch


def _mlp(
    in_dim: int,
    hidden: tuple[int, ...],
    out_dim: int,
    *,
    batch_norm: bool = False,
) -> Any:
    torch = require_torch(feature="SSL models")
    layers: list[Any] = []
    prev = in_dim
    for width in hidden:
        layers.append(torch.nn.Linear(prev, width))
        if batch_norm:
            layers.append(torch.nn.BatchNorm1d(width))
        layers.append(torch.nn.ReLU(inplace=True))
        prev = width
    layers.append(torch.nn.Linear(prev, out_dim))
    return torch.nn.Sequential(*layers)


def build_tabular_encoder(
    n_features: int,
    *,
    hidden: tuple[int, ...],
    latent_dim: int,
) -> Any:
    torch = require_torch(feature="SSL models")

    class TabularEncoderModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.n_features = int(n_features)
            self.latent_dim = int(latent_dim)
            self.net = _mlp(n_features, hidden, latent_dim)

        def forward(self, x: Any) -> Any:
            return self.net(x)

    return TabularEncoderModule()


def build_projector(latent_dim: int, *, hidden: tuple[int, ...], out_dim: int) -> Any:
    return _mlp(latent_dim, hidden, out_dim)


def build_predictor(in_dim: int, *, hidden: tuple[int, ...], out_dim: int) -> Any:
    """BYOL online predictor (projector_dim → projector_dim)."""
    return _mlp(in_dim, hidden, out_dim)


def build_mae_decoder(latent_dim: int, *, hidden: tuple[int, ...], n_features: int) -> Any:
    torch = require_torch(feature="SSL models")

    class MAEDecoderModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.decoder = _mlp(latent_dim, hidden, n_features)

        def forward(self, z: Any) -> Any:
            return self.decoder(z)

    return MAEDecoderModule()


def build_vae(n_features: int, *, hidden: tuple[int, ...], latent_dim: int) -> Any:
    torch = require_torch(feature="SSL models")
    enc_hidden = hidden if hidden else (max(n_features, latent_dim * 2),)
    last = enc_hidden[-1]

    class VAEModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.n_features = int(n_features)
            self.latent_dim = int(latent_dim)
            self.encoder_body = _mlp(n_features, enc_hidden, last)
            self.mu = torch.nn.Linear(last, latent_dim)
            self.logvar = torch.nn.Linear(last, latent_dim)
            self.decoder = _mlp(latent_dim, enc_hidden, n_features)

        def encode(self, x: Any) -> tuple[Any, Any]:
            h = self.encoder_body(x)
            return self.mu(h), self.logvar(h)

        def reparameterize(self, mu: Any, logvar: Any) -> Any:
            if self.training:
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                return mu + eps * std
            return mu

        def forward(self, x: Any) -> tuple[Any, Any, Any, Any]:
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            recon = self.decoder(z)
            return recon, mu, logvar, z

    return VAEModule()


def build_simclr_module(
    n_features: int,
    *,
    hidden: tuple[int, ...],
    latent_dim: int,
    projector_hidden: tuple[int, ...],
    projector_dim: int,
) -> Any:
    torch = require_torch(feature="SSL SimCLR")
    encoder = build_tabular_encoder(n_features, hidden=hidden, latent_dim=latent_dim)
    projector = build_projector(latent_dim, hidden=projector_hidden, out_dim=projector_dim)

    class SimCLR(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = encoder
            self.projector = projector

        def forward(self, x: Any) -> tuple[Any, Any]:
            z = self.encoder(x)
            p = self.projector(z)
            return z, p

    return SimCLR()


def build_byol_module(
    n_features: int,
    *,
    hidden: tuple[int, ...],
    latent_dim: int,
    projector_hidden: tuple[int, ...],
    projector_dim: int,
    predictor_hidden: tuple[int, ...],
) -> tuple[Any, Any]:
    """Return (online_module, target_module) for BYOL."""
    torch = require_torch(feature="SSL BYOL")

    class BYOLNet(torch.nn.Module):
        def __init__(
            self,
            *,
            with_predictor: bool,
            copy_from: Any | None = None,
        ) -> None:
            super().__init__()
            if copy_from is not None:
                self.encoder = build_tabular_encoder(
                    n_features, hidden=hidden, latent_dim=latent_dim
                )
                self.projector = build_projector(
                    latent_dim, hidden=projector_hidden, out_dim=projector_dim
                )
                self.predictor = None
                return
            self.encoder = build_tabular_encoder(
                n_features, hidden=hidden, latent_dim=latent_dim
            )
            self.projector = build_projector(
                latent_dim, hidden=projector_hidden, out_dim=projector_dim
            )
            self.predictor = (
                build_predictor(projector_dim, hidden=predictor_hidden, out_dim=projector_dim)
                if with_predictor
                else None
            )

        def forward(self, x: Any) -> tuple[Any, Any]:
            z = self.encoder(x)
            p = self.projector(z)
            if self.predictor is not None:
                p = self.predictor(p)
            return z, p

    online = BYOLNet(with_predictor=True)
    target = BYOLNet(with_predictor=False)
    target.load_state_dict(
        {
            k: v
            for k, v in online.state_dict().items()
            if not k.startswith("predictor.")
        },
        strict=False,
    )
    for param in target.parameters():
        param.requires_grad = False
    return online, target


def build_vicreg_module(
    n_features: int,
    *,
    hidden: tuple[int, ...],
    latent_dim: int,
    projector_hidden: tuple[int, ...],
    projector_dim: int,
) -> Any:
    return build_simclr_module(
        n_features,
        hidden=hidden,
        latent_dim=latent_dim,
        projector_hidden=projector_hidden,
        projector_dim=projector_dim,
    )


def build_mae_module(
    n_features: int,
    *,
    hidden: tuple[int, ...],
    latent_dim: int,
) -> Any:
    torch = require_torch(feature="SSL MAE")
    encoder = build_tabular_encoder(n_features, hidden=hidden, latent_dim=latent_dim)
    decoder = build_mae_decoder(latent_dim, hidden=hidden, n_features=n_features)

    class MAE(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder

        def forward(self, x: Any) -> tuple[Any, Any]:
            z = self.encoder(x)
            recon = self.decoder(z)
            return z, recon

    return MAE()
