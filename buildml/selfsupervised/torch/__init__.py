"""Torch SSL subpackage (lazy — requires buildml[torch] / buildml[ssl])."""

from __future__ import annotations

from typing import Any

__all__ = [
    "HFTextSSLEncoder",
    "TorchTabularSSLEncoder",
    "VisionSSLEncoder",
    "list_ssl_methods",
    "resolve_default_tabular_method",
    "train_tabular_ssl",
]


def __getattr__(name: str) -> Any:
    if name in {"TorchTabularSSLEncoder", "train_tabular_ssl"}:
        from buildml.selfsupervised.torch.encoder import TorchTabularSSLEncoder
        from buildml.selfsupervised.torch.trainer import train_tabular_ssl

        return {"TorchTabularSSLEncoder": TorchTabularSSLEncoder, "train_tabular_ssl": train_tabular_ssl}[name]
    if name == "HFTextSSLEncoder":
        from buildml.selfsupervised.torch.text import HFTextSSLEncoder

        return HFTextSSLEncoder
    if name == "VisionSSLEncoder":
        from buildml.selfsupervised.torch.vision import VisionSSLEncoder

        return VisionSSLEncoder
    if name in {"list_ssl_methods", "resolve_default_tabular_method"}:
        from buildml.selfsupervised.torch import catalog as catalog_mod

        return getattr(catalog_mod, name)
    raise AttributeError(f"module 'buildml.selfsupervised.torch' has no attribute {name!r}")
