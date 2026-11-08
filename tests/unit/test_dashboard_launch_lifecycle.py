"""Dashboard launch handles own only the server they started."""

import socket
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from threading import Event
from urllib.request import ProxyHandler, build_opener

import pandas as pd
import pytest

from buildml import Session
from buildml.dashboard.launch import DashboardLaunchError, launch_eda_app
from buildml.dashboard.state import get_state


@pytest.fixture
def reports():
    pytest.importorskip("fastapi")
    pytest.importorskip("uvicorn")
    pytest.importorskip("jinja2")
    return [
        Session.ingest(pd.DataFrame({name: [1, 2, 4, 8]})).eda(include_plots=False, show=False)
        for name in ("original_measure", "replacement_measure")
    ]


def _available_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def _home(url):
    with build_opener(ProxyHandler({})).open(url, timeout=5) as response:
        assert response.status == 200
        return response.read().decode("utf-8")


@pytest.mark.parametrize("stop_before_replacement", [False, True])
def test_old_handle_cannot_clear_replacement_state(reports, stop_before_replacement):
    original, replacement = reports
    first = launch_eda_app(
        original, port=_available_port(), title="Original studio", open_browser=False
    )
    try:
        assert "Original studio" in _home(first.url)
        if stop_before_replacement:
            first.stop()
            assert not first.is_running
        second = launch_eda_app(
            replacement, port=_available_port(), title="Replacement studio", open_browser=False
        )
        try:
            assert "Replacement studio" in _home(second.url)
            first.stop()
            assert not first.is_running
            assert second.is_running
            assert "Replacement studio" in _home(second.url)
        finally:
            second.stop()
        assert not second.is_running
        with pytest.raises(RuntimeError, match="state is not initialized"):
            get_state()
    finally:
        first.stop()


def test_invalid_bind_address_reports_startup_failure(reports):
    port = _available_port()
    host = "127.0.0.1/24"
    with pytest.raises(DashboardLaunchError, match=f"failed to start.*{port}") as exc_info:
        launch_eda_app(reports[0], host=host, port=port, open_browser=False)
    assert host in str(exc_info.value)
    assert "already in use" not in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, OSError)


def test_completed_blocking_handle_reports_stopped_and_can_be_stopped_again(reports, monkeypatch):
    import uvicorn

    servers = Queue()
    shutdown = Event()
    server = None
    server_class = uvicorn.Server
    port = _available_port()

    def capture_server(config):
        nonlocal server
        server = server_class(config)
        server.should_exit = shutdown.is_set()
        servers.put(server)
        return server

    def launch():
        handle = launch_eda_app(reports[0], port=port, open_browser=False, blocking=True)
        assert not handle.is_running
        handle.stop()
        return handle

    monkeypatch.setattr(uvicorn, "Server", capture_server)
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(launch)
        try:
            server = servers.get(timeout=10)
            # Exercise the real Uvicorn server without opening a browser.
            deadline = time.monotonic() + 10
            while not server.started and not result.done() and time.monotonic() < deadline:
                time.sleep(0.025)
            assert server.started
            assert "BuildML EDA Studio" in _home(f"http://127.0.0.1:{port}/")
        finally:
            shutdown.set()
            if server is not None:
                server.should_exit = True
        handle = result.result(timeout=10)
    assert not handle.is_running
    with pytest.raises(RuntimeError, match="state is not initialized"):
        get_state()
