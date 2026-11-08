"""Serve real bundles over loopback and preserve startup failure diagnostics."""

from __future__ import annotations

import socket
import ssl
from datetime import datetime, timedelta, timezone
from ipaddress import IPv4Address

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from buildml import Session
from buildml.serving.config import ServeConfig
from buildml.serving.launch import ServingLaunchError, serve_bundle


@pytest.fixture
def bundle(tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("uvicorn")
    frame = pd.DataFrame({"x": np.arange(40, dtype=float)})
    frame["y"] = 2 * frame.x + 3
    session = Session.ingest(frame).set_roles({"x": "feature", "y": "target"})
    session.split(test_size=0.2, random_state=4)
    session.fit(LinearRegression(), task="regression")
    return session.save_pipeline(tmp_path / "pipeline")


def _bind_loopback(family):
    address = "::1" if family == socket.AF_INET6 else "127.0.0.1"
    listener = None
    try:
        listener = socket.socket(family, socket.SOCK_STREAM)
        listener.bind((address, 0))
    except OSError as exc:
        if listener is not None:
            listener.close()
        if family == socket.AF_INET6:
            pytest.skip(f"IPv6 loopback unavailable: {exc}")
        raise
    return listener


def _free_port(family=socket.AF_INET):
    with _bind_loopback(family) as listener:
        return listener.getsockname()[1]


@pytest.mark.parametrize("host", ["::1", "[::1]"])
def test_ipv6_handle_url_serves_real_predictions(bundle, host):
    httpx = pytest.importorskip("httpx")
    port = _free_port(socket.AF_INET6)
    config = ServeConfig(bundle=str(bundle), host=host, port=port, trusted=True)
    handle = serve_bundle(config=config)
    try:
        assert handle.url == f"http://[::1]:{port}"
        with httpx.Client(base_url=handle.url, timeout=5, trust_env=False) as client:
            response = client.post("/predict", json={"rows": [{"x": 4.0}, {"x": -7.0}]})
            response.raise_for_status()
            np.testing.assert_allclose(response.json()["predictions"], [11.0, -11.0])
    finally:
        handle.stop()
    assert not handle.is_running
    with socket.socket(socket.AF_INET6) as connection:
        assert connection.connect_ex(("::1", port)) != 0


def test_occupied_ipv6_loopback_is_detected_before_artifact_load(tmp_path):
    pytest.importorskip("uvicorn")
    with _bind_loopback(socket.AF_INET6) as listener:
        listener.listen()
        port = listener.getsockname()[1]
        with pytest.raises(ServingLaunchError, match=f"Cannot bind.*{port}") as captured:
            serve_bundle(tmp_path / "missing-artifact", host="::1", port=port, trusted=True)
    assert isinstance(captured.value.__cause__, OSError)


@pytest.mark.parametrize("blocking", [False, True])
def test_bad_tls_reports_the_real_startup_error(bundle, tmp_path, blocking):
    cert, key = tmp_path / "bad-cert.pem", tmp_path / "bad-key.pem"
    cert.write_text("not a certificate", encoding="ascii")
    key.write_text("not a private key", encoding="ascii")
    port = _free_port()
    with pytest.raises(ServingLaunchError, match=f"startup.*{port}") as captured:
        serve_bundle(
            bundle,
            port=port,
            ssl_certfile=cert,
            ssl_keyfile=key,
            trusted=True,
            blocking=blocking,
        )
    assert isinstance(captured.value.__cause__, ssl.SSLError)
    with socket.socket() as connection:
        assert connection.connect_ex(("127.0.0.1", port)) != 0


def test_real_https_preserves_auth_and_prediction(bundle, tmp_path):
    httpx = pytest.importorskip("httpx")
    pytest.importorskip("cryptography")
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "loopback test")])
    now = datetime.now(timezone.utc)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(hours=1))
        .add_extension(
            x509.SubjectAlternativeName([x509.IPAddress(IPv4Address("127.0.0.1"))]), False
        )
        .sign(private_key, hashes.SHA256())
    )
    cert, key = tmp_path / "loopback-cert.pem", tmp_path / "loopback-key.pem"
    cert.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    key.write_bytes(
        private_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )

    port = _free_port()
    handle = serve_bundle(
        bundle,
        port=port,
        ssl_certfile=cert,
        ssl_keyfile=key,
        api_keys="test-loopback-key",
        trusted=True,
    )
    try:
        assert handle.tls and handle.url == f"https://127.0.0.1:{port}"
        context = ssl.create_default_context(cafile=str(cert))
        with httpx.Client(
            base_url=handle.url, verify=context, timeout=5, trust_env=False
        ) as client:
            payload = {"rows": [{"x": 4.0}]}
            assert client.post("/predict", json=payload).status_code == 401
            response = client.post(
                "/predict", headers={"X-API-Key": "test-loopback-key"}, json=payload
            )
            response.raise_for_status()
            np.testing.assert_allclose(response.json()["predictions"], [11.0])
    finally:
        handle.stop()
    assert not handle.is_running
