"""Local EDA Teaching Studio app (FastAPI + SPA shell)."""

from buildml.dashboard.launch import (
    DashboardLaunchError,
    EDAAppHandle,
    launch_eda_app,
    open_eda_dashboard,
)
from buildml.dashboard.offline import export_studio_html

__all__ = [
    "DashboardLaunchError",
    "EDAAppHandle",
    "export_studio_html",
    "launch_eda_app",
    "open_eda_dashboard",
]
