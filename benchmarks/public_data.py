"""Explicit cached public-data downloads with reproducible source fingerprints."""
from __future__ import annotations

import hashlib
import ssl
import urllib.request
from pathlib import Path


def public_file(url: str, destination: Path, *, download: bool) -> dict[str, str]:
    """Fetch only when requested; fail rather than substitute synthetic data."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        if not download:
            raise FileNotFoundError(f"Missing {destination}; run with --download to fetch {url}")
        temporary = destination.with_suffix(destination.suffix + ".part")
        try:
            # Use OS certificate trust where available; never disable TLS verification.
            try:
                import truststore
                context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            except ImportError:
                context = ssl.create_default_context()
            with urllib.request.urlopen(url, timeout=60, context=context) as response, temporary.open("wb") as output:
                while block := response.read(1024 * 1024):
                    output.write(block)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    return {"url": url, "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            "path": str(destination), "fallback": "none"}
