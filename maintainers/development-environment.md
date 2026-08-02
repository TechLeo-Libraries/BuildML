# Development environment

BuildML 2.x development uses a project-local virtual environment so dependency pins stay reproducible.

## Create / activate

```powershell
cd "C:\Users\leona\Desktop\Github Projects\BuildML"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

`.venv/` is gitignored.

## Refresh pinned requirements from the venv

Core runtime pins (no editable project, no dev tools):

```powershell
.\.venv\Scripts\Activate.ps1
pip install -e .
pip freeze --exclude-editable > requirements.txt
```

Dev pins (tests/lint/build tools included):

```powershell
pip install -e ".[dev]"
pip freeze --exclude-editable > requirements-dev.txt
```

Canonical dependency declarations remain in `pyproject.toml`. The requirements files are frozen snapshots for clean installs and audits.

## Run checks

```powershell
pytest
ruff check buildml tests
```
